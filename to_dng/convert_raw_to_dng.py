from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

try:
    import tifffile
except ImportError:
    tifffile = None

try:
    import rawpy
except ImportError:
    rawpy = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector
except ImportError:
    plt = None
    RectangleSelector = None


COMMON_ASPECT_RATIOS = (4 / 3, 5 / 4, 3 / 2, 16 / 9, 1.0)
PATTERN_ALIASES = {
    "BAYRG8": "RGGB",
    "BAYGR8": "GRBG",
    "BAYGRB": "GRBG",
    "BAYBG8": "BGGR",
    "BAYGB8": "GBRG",
    "RGGB": "RGGB",
    "GRBG": "GRBG",
    "BGGR": "BGGR",
    "GBRG": "GBRG",
}
CFA_PATTERN_BYTES = {
    "RGGB": bytes([0, 1, 1, 2]),
    "GRBG": bytes([1, 0, 2, 1]),
    "BGGR": bytes([2, 1, 1, 0]),
    "GBRG": bytes([1, 2, 0, 1]),
}
SIDECAR_SUFFIXES = (".json", ".meta.json", ".metadata.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an 8-bit Bayer RAW buffer into a minimal uncompressed DNG."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent,
        help="Input RAW file path or input directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("output"),
        help="Output DNG file path (single input) or output directory (batch).",
    )
    parser.add_argument(
        "--input-glob",
        default="*.raw",
        help="Glob pattern used when --input is a directory.",
    )
    parser.add_argument(
        "--pattern",
        default="BAYGRB",
        help="Bayer pattern, e.g. BAYGRB/BAYGR8/GRBG.",
    )
    parser.add_argument(
        "--conversion",
        default="LINE",
        choices=("RAW", "LINE", "raw", "line"),
        help="RAW: file size must equal width*height; LINE: take first width*height bytes and ignore tail padding.",
    )
    parser.add_argument("--width", type=int, help="Image width in pixels. LINE mode default: 2592.")
    parser.add_argument("--height", type=int, help="Image height in pixels. LINE mode default: 1944.")
    parser.add_argument(
        "--white-level",
        type=int,
        default=255,
        help="DNG WhiteLevel tag value.",
    )
    parser.add_argument(
        "--black-level",
        type=int,
        default=0,
        help="Per-channel black level written into the DNG.",
    )
    parser.add_argument(
        "--make",
        default="OpenAI",
        help="TIFF Make tag.",
    )
    parser.add_argument(
        "--model",
        default="BAYGR8 Converter",
        help="TIFF Model tag.",
    )
    parser.add_argument(
        "--exposure-unit",
        default="ms",
        choices=("us", "ms", "s"),
        help="Fallback unit used when exposure is inferred from numeric filename or unit-less metadata.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip rawpy re-open validation after writing the DNG.",
    )
    parser.add_argument(
        "--select-roi",
        action="store_true",
        help="Interactively select an ROI on raw intensity preview.",
    )
    parser.add_argument(
        "--roi",
        help="Manual ROI as x,y,w,h in pixel coordinates.",
    )
    parser.add_argument(
        "--roi-output",
        type=Path,
        help="Path to save ROI JSON. Default: <input_stem>.roi.json next to input.",
    )
    parser.add_argument(
        "--roi-only",
        action="store_true",
        help="Run ROI selection/stats only and skip DNG writing.",
    )
    return parser.parse_args()


def normalize_pattern(pattern: str) -> str:
    normalized = pattern.strip().upper()
    if normalized not in PATTERN_ALIASES:
        supported = ", ".join(sorted(PATTERN_ALIASES))
        raise ValueError(f"Unsupported Bayer pattern '{pattern}'. Supported: {supported}")
    return PATTERN_ALIASES[normalized]


def normalize_conversion(conversion: str) -> str:
    normalized = conversion.strip().upper()
    if normalized not in {"RAW", "LINE"}:
        raise ValueError(f"Unsupported conversion '{conversion}'. Supported: RAW, LINE")
    return normalized


def list_raw_inputs(input_path: Path, input_glob: str) -> list[Path]:
    expanded = input_path.expanduser()
    if expanded.is_file():
        return [expanded]
    if not expanded.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {expanded}")

    files = sorted(path for path in expanded.glob(input_glob) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No RAW files matched '{input_glob}' in {expanded}")
    return files


def resolve_output_path(input_path: Path, output_target: Path, multiple_inputs: bool) -> Path:
    expanded = output_target.expanduser()
    if expanded.suffix.lower() == ".dng":
        if multiple_inputs:
            raise ValueError(
                "When processing multiple RAW files, --output must be a directory, not a single .dng file."
            )
        return expanded
    return expanded / f"{input_path.stem}.dng"


def infer_dimensions_for_input(
    raw_bytes: np.ndarray,
    conversion: str,
    width_arg: int | None,
    height_arg: int | None,
) -> tuple[int, int]:
    if (width_arg is None) ^ (height_arg is None):
        raise ValueError("Please provide both --width and --height, or neither.")
    if width_arg is not None and height_arg is not None:
        return width_arg, height_arg
    if conversion == "LINE":
        return 2592, 1944
    return infer_dimensions(raw_bytes)


def parse_fraction_or_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if "/" in text:
        parts = [part.strip() for part in text.split("/", maxsplit=1)]
        if len(parts) == 2:
            try:
                numerator = float(parts[0])
                denominator = float(parts[1])
                if denominator != 0:
                    return numerator / denominator
            except ValueError:
                return None
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_int_value(value: Any) -> int | None:
    parsed = parse_fraction_or_float(value)
    if parsed is None:
        return None
    int_value = int(round(parsed))
    if int_value <= 0:
        return None
    return int_value


def float_to_rational(value: float, max_denominator: int = 1_000_000) -> tuple[int, int]:
    frac = Fraction(value).limit_denominator(max_denominator)
    return int(frac.numerator), int(frac.denominator)


def parse_datetime_for_exif(value: Any) -> str | None:
    if isinstance(value, (int, float)) and float(value) > 0:
        timestamp = float(value)
        if timestamp > 1e12:
            timestamp /= 1_000.0
        try:
            parsed = datetime.fromtimestamp(timestamp)
        except (OverflowError, OSError, ValueError):
            return None
        return parsed.strftime("%Y:%m:%d %H:%M:%S")

    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if re.fullmatch(r"\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}", text):
        return text
    normalized = text.replace("T", " ").replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return parsed.strftime("%Y:%m:%d %H:%M:%S")


def convert_exposure_to_seconds(value: float, unit: str) -> float | None:
    if value <= 0:
        return None
    if unit == "s":
        return value
    if unit == "ms":
        return value / 1_000.0
    if unit == "us":
        return value / 1_000_000.0
    return None


def parse_exposure_seconds(value: Any, default_unit: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return convert_exposure_to_seconds(float(value), default_unit)
    if not isinstance(value, str):
        return None

    text = value.strip().lower()
    if not text:
        return None
    if "/" in text:
        parsed = parse_fraction_or_float(text)
        if parsed is None:
            return None
        return convert_exposure_to_seconds(parsed, "s")

    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)(?:\s*(us|µs|ms|s|sec|secs|second|seconds))?", text)
    if not match:
        return None
    numeric = float(match.group(1))
    unit_token = match.group(2)
    if unit_token in {"s", "sec", "secs", "second", "seconds"}:
        unit = "s"
    elif unit_token in {"us", "µs"}:
        unit = "us"
    elif unit_token == "ms":
        unit = "ms"
    else:
        unit = default_unit
    return convert_exposure_to_seconds(numeric, unit)


def parse_exposure_from_filename(stem: str, default_unit: str) -> float | None:
    direct = parse_exposure_seconds(stem, default_unit)
    if direct is not None:
        return direct
    match = re.search(r"([+-]?\d+(?:\.\d+)?)", stem)
    if not match:
        return None
    try:
        numeric = float(match.group(1))
    except ValueError:
        return None
    return convert_exposure_to_seconds(numeric, default_unit)


def find_first_metadata_value(payload: Any, keys: tuple[str, ...]) -> Any | None:
    keyset = {key.lower() for key in keys}
    stack: list[Any] = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for raw_key, raw_value in item.items():
                key = str(raw_key).strip().lower()
                if key in keyset and not isinstance(raw_value, (dict, list)):
                    return raw_value
                if isinstance(raw_value, (dict, list)):
                    stack.append(raw_value)
        elif isinstance(item, list):
            stack.extend(item)
    return None


def load_sidecar_payload(raw_path: Path) -> tuple[dict[str, Any] | list[Any] | None, Path | None]:
    for suffix in SIDECAR_SUFFIXES:
        sidecar_path = raw_path.with_suffix(suffix)
        if not sidecar_path.exists():
            continue
        try:
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: failed to parse metadata sidecar {sidecar_path.name}: {exc}")
            continue
        if isinstance(payload, (dict, list)):
            return payload, sidecar_path
    return None, None


def collect_capture_metadata(
    input_path: Path,
    default_make: str,
    default_model: str,
    exposure_unit: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "make": default_make,
        "model": default_model,
        "source": "defaults",
    }
    payload, sidecar_path = load_sidecar_payload(input_path)
    if payload is not None:
        metadata["source"] = str(sidecar_path) if sidecar_path is not None else "sidecar"
        sidecar_make = find_first_metadata_value(payload, ("make", "camera_make", "manufacturer"))
        sidecar_model = find_first_metadata_value(payload, ("model", "camera_model", "device"))
        sidecar_iso = find_first_metadata_value(
            payload,
            ("iso", "iso_speed", "iso_value", "sensitivity", "photographicsensitivity"),
        )
        sidecar_f_number = find_first_metadata_value(payload, ("f_number", "fnumber", "aperture", "fstop"))
        sidecar_time = find_first_metadata_value(
            payload,
            ("datetime", "date_time", "datetimeoriginal", "capture_time", "timestamp", "time"),
        )
        sidecar_exposure_seconds = find_first_metadata_value(payload, ("exposure_time", "exposure_seconds"))
        sidecar_exposure_ms = find_first_metadata_value(payload, ("exposure_ms",))
        sidecar_exposure_us = find_first_metadata_value(payload, ("exposure_us",))
        sidecar_exposure_generic = find_first_metadata_value(payload, ("exposure", "shutter", "shutter_speed"))

        if isinstance(sidecar_make, str) and sidecar_make.strip():
            metadata["make"] = sidecar_make.strip()
        if isinstance(sidecar_model, str) and sidecar_model.strip():
            metadata["model"] = sidecar_model.strip()

        exposure_seconds = (
            parse_exposure_seconds(sidecar_exposure_seconds, "s")
            or parse_exposure_seconds(sidecar_exposure_ms, "ms")
            or parse_exposure_seconds(sidecar_exposure_us, "us")
            or parse_exposure_seconds(sidecar_exposure_generic, exposure_unit)
        )
        if exposure_seconds is not None:
            metadata["exposure_time"] = exposure_seconds
        iso_value = parse_int_value(sidecar_iso)
        if iso_value is not None:
            metadata["iso"] = min(iso_value, 65535)
        f_number = parse_fraction_or_float(sidecar_f_number)
        if f_number is not None and f_number > 0:
            metadata["f_number"] = float(f_number)
        date_time = parse_datetime_for_exif(sidecar_time)
        if date_time is not None:
            metadata["date_time"] = date_time

    if "exposure_time" not in metadata:
        inferred = parse_exposure_from_filename(input_path.stem, exposure_unit)
        if inferred is not None:
            metadata["exposure_time"] = inferred
            if metadata["source"] == "defaults":
                metadata["source"] = "filename"
    return metadata


def format_metadata_summary(metadata: dict[str, Any]) -> str:
    chunks: list[str] = [f"source={metadata.get('source', 'unknown')}"]
    if "exposure_time" in metadata:
        chunks.append(f"exposure={float(metadata['exposure_time']):.6f}s")
    if "iso" in metadata:
        chunks.append(f"iso={int(metadata['iso'])}")
    if "f_number" in metadata:
        chunks.append(f"f/{float(metadata['f_number']):.2f}")
    if "date_time" in metadata:
        chunks.append(f"time={metadata['date_time']}")
    return ", ".join(chunks)


def even_factor_pairs(pixel_count: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for height in range(2, math.isqrt(pixel_count) + 1):
        if pixel_count % height != 0:
            continue
        width = pixel_count // height
        if width % 2 == 0 and height % 2 == 0:
            candidates.append((width, height))
    return candidates


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("-inf")
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return float("-inf")
    return float(np.corrcoef(a, b)[0, 1])


def candidate_score(raw_bytes: np.ndarray, width: int, height: int) -> float:
    image = raw_bytes.reshape(height, width).astype(np.float32)
    row_corr = safe_corrcoef(image[:-1].ravel(), image[1:].ravel())
    even_row_corr = safe_corrcoef(image[0:-2:2].ravel(), image[2::2].ravel())
    aspect_ratio = width / height
    aspect_penalty = min(abs(aspect_ratio - ratio) for ratio in COMMON_ASPECT_RATIOS)
    return (row_corr * 2.0) + even_row_corr - (aspect_penalty * 0.25)


def infer_dimensions(raw_bytes: np.ndarray) -> tuple[int, int]:
    candidates = even_factor_pairs(raw_bytes.size)
    if not candidates:
        raise ValueError("Unable to infer even Bayer dimensions from RAW byte count.")
    ranked = sorted(
        ((candidate_score(raw_bytes, width, height), width, height) for width, height in candidates),
        reverse=True,
    )
    _, width, height = ranked[0]
    return width, height


def build_extratags(
    pattern: str,
    black_level: int,
    white_level: int,
    unique_camera_model: str,
    make: str,
    model: str,
    capture_metadata: dict[str, Any] | None = None,
) -> list[tuple[int, tifffile.DATATYPE, int, object, bool]]:
    if tifffile is None:
        raise RuntimeError("DNG writing requires tifffile. Install it or use --roi-only.")
    black_levels = (black_level, black_level, black_level, black_level)
    black_level_rational = []
    for value in black_levels:
        black_level_rational.extend((int(value), 1))

    extratags: list[tuple[int, tifffile.DATATYPE, int, object, bool]] = [
        (271, tifffile.DATATYPE.ASCII, 1, make, False),
        (272, tifffile.DATATYPE.ASCII, 1, model, False),
        (274, tifffile.DATATYPE.SHORT, 1, 1, False),
        (33421, tifffile.DATATYPE.SHORT, 2, (2, 2), False),
        (33422, tifffile.DATATYPE.BYTE, 4, CFA_PATTERN_BYTES[pattern], False),
        (50706, tifffile.DATATYPE.BYTE, 4, bytes([1, 4, 0, 0]), False),
        (50707, tifffile.DATATYPE.BYTE, 4, bytes([1, 1, 0, 0]), False),
        (50708, tifffile.DATATYPE.ASCII, 1, unique_camera_model, False),
        (50710, tifffile.DATATYPE.BYTE, 3, bytes([0, 1, 2]), False),
        (50711, tifffile.DATATYPE.SHORT, 1, 1, False),
        (50713, tifffile.DATATYPE.SHORT, 2, (2, 2), False),
        (50714, tifffile.DATATYPE.RATIONAL, 4, tuple(black_level_rational), False),
        (50717, tifffile.DATATYPE.LONG, 1, int(white_level), False),
    ]

    if capture_metadata:
        exposure_time = capture_metadata.get("exposure_time")
        if isinstance(exposure_time, (int, float)) and float(exposure_time) > 0:
            extratags.append(
                (33434, tifffile.DATATYPE.RATIONAL, 1, float_to_rational(float(exposure_time)), False)
            )

        f_number = capture_metadata.get("f_number")
        if isinstance(f_number, (int, float)) and float(f_number) > 0:
            extratags.append((33437, tifffile.DATATYPE.RATIONAL, 1, float_to_rational(float(f_number)), False))

        iso = capture_metadata.get("iso")
        if isinstance(iso, int) and iso > 0:
            extratags.append((34855, tifffile.DATATYPE.SHORT, 1, min(iso, 65535), False))

        date_time = capture_metadata.get("date_time")
        if isinstance(date_time, str) and date_time:
            extratags.extend(
                [
                    (306, tifffile.DATATYPE.ASCII, 1, date_time, False),
                    (36867, tifffile.DATATYPE.ASCII, 1, date_time, False),
                    (36868, tifffile.DATATYPE.ASCII, 1, date_time, False),
                ]
            )

    return extratags


def decode_raw_plane(
    raw_bytes: np.ndarray,
    width: int,
    height: int,
    conversion: str,
) -> tuple[np.ndarray, int]:
    expected_size = width * height
    if conversion == "RAW":
        if raw_bytes.size != expected_size:
            raise ValueError(
                f"RAW size mismatch: file has {raw_bytes.size} bytes, but {width}x{height} expects {expected_size}."
            )
        return raw_bytes.reshape(height, width), 0

    if raw_bytes.size < expected_size:
        raise ValueError(
            f"LINE size mismatch: file has {raw_bytes.size} bytes, but at least {expected_size} are required for "
            f"{width}x{height}."
        )

    payload = raw_bytes[:expected_size]
    trailing = raw_bytes[expected_size:]
    if trailing.size and np.any(trailing != 0):
        non_zero = int(np.count_nonzero(trailing))
        print(
            f"Warning: LINE tail has {non_zero}/{trailing.size} non-zero bytes; "
            "truncating to the first frame anyway."
        )
    return payload.reshape(height, width), int(trailing.size)


def parse_roi_spec(spec: str) -> tuple[int, int, int, int]:
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be in format x,y,w,h")
    x, y, w, h = (int(p) for p in parts)
    return x, y, w, h


def clip_roi(roi: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        raise ValueError("ROI width and height must be positive")
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = min(w, width - x)
    h = min(h, height - y)
    if w <= 0 or h <= 0:
        raise ValueError("ROI is outside image bounds")
    return x, y, w, h


def preview_image(image: np.ndarray) -> np.ndarray:
    p_low, p_high = np.percentile(image, (1, 99))
    if p_high <= p_low:
        return np.zeros_like(image, dtype=np.float32)
    scaled = (image.astype(np.float32) - float(p_low)) / float(p_high - p_low)
    return np.clip(scaled, 0.0, 1.0)


def select_roi_interactive(image: np.ndarray) -> tuple[int, int, int, int]:
    if plt is None or RectangleSelector is None:
        raise RuntimeError("Interactive ROI requires matplotlib. Install it or use --roi x,y,w,h.")

    state: dict[str, tuple[int, int, int, int] | None] = {"roi": None}
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(preview_image(image), cmap="gray", interpolation="nearest")
    ax.set_title("Drag to select ROI, then press Enter to confirm (Esc to cancel)")

    def on_select(eclick, erelease) -> None:
        if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
            return
        x0 = int(round(min(eclick.xdata, erelease.xdata)))
        y0 = int(round(min(eclick.ydata, erelease.ydata)))
        x1 = int(round(max(eclick.xdata, erelease.xdata)))
        y1 = int(round(max(eclick.ydata, erelease.ydata)))
        state["roi"] = (x0, y0, max(1, x1 - x0 + 1), max(1, y1 - y0 + 1))

    def on_key(event) -> None:
        key = (event.key or "").lower()
        if key in {"enter", "return"}:
            plt.close(fig)
        elif key == "escape":
            state["roi"] = None
            plt.close(fig)

    selector = RectangleSelector(
        ax,
        on_select,
        useblit=True,
        button=[1],
        minspanx=2,
        minspany=2,
        spancoords="pixels",
        interactive=True,
        drag_from_anywhere=True,
    )
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    if state["roi"] is None:
        raise RuntimeError("ROI selection canceled or no ROI selected")

    selector.set_active(False)
    return state["roi"]


def roi_stats(image: np.ndarray, roi: tuple[int, int, int, int]) -> dict[str, float | int]:
    x, y, w, h = roi
    patch = image[y : y + h, x : x + w].astype(np.float32)
    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "pixels": int(patch.size),
        "mean": float(np.mean(patch)),
        "std": float(np.std(patch)),
        "min": float(np.min(patch)),
        "max": float(np.max(patch)),
    }


def default_roi_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.roi.json")


def save_roi(
    output_path: Path,
    input_path: Path,
    width: int,
    height: int,
    conversion: str,
    pattern: str,
    stats: dict[str, float | int],
) -> None:
    payload = {
        "input": str(input_path),
        "width": width,
        "height": height,
        "conversion": conversion,
        "pattern": pattern,
        "roi": {
            "x": int(stats["x"]),
            "y": int(stats["y"]),
            "w": int(stats["w"]),
            "h": int(stats["h"]),
        },
        "stats": {
            "pixels": int(stats["pixels"]),
            "mean": float(stats["mean"]),
            "std": float(stats["std"]),
            "min": float(stats["min"]),
            "max": float(stats["max"]),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_dng(
    input_path: Path,
    output_path: Path,
    width: int,
    height: int,
    conversion: str,
    pattern: str,
    black_level: int,
    white_level: int,
    make: str,
    model: str,
    capture_metadata: dict[str, Any] | None = None,
) -> None:
    if tifffile is None:
        raise RuntimeError("DNG writing requires tifffile. Install it or use --roi-only.")

    raw_bytes = np.fromfile(input_path, dtype=np.uint8)
    image, trailing_bytes = decode_raw_plane(
        raw_bytes=raw_bytes,
        width=width,
        height=height,
        conversion=conversion,
    )
    if conversion == "LINE" and trailing_bytes:
        print(f"LINE conversion: ignored {trailing_bytes} trailing padding bytes.")
    effective_make = str(capture_metadata.get("make", make)) if capture_metadata else make
    effective_model = str(capture_metadata.get("model", model)) if capture_metadata else model
    unique_camera_model = f"{effective_model}-{pattern}-{width}x{height}"
    extratags = build_extratags(
        pattern=pattern,
        black_level=black_level,
        white_level=white_level,
        unique_camera_model=unique_camera_model,
        make=effective_make,
        model=effective_model,
        capture_metadata=capture_metadata,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffWriter(output_path, bigtiff=False) as tif:
        tif.write(
            image,
            dtype=np.uint8,
            photometric=tifffile.PHOTOMETRIC.CFA,
            bitspersample=8,
            compression=0,
            metadata=None,
            subfiletype=0,
            software="OpenAI Codex CLI",
            extratags=extratags,
        )


def validate_dng(output_path: Path) -> None:
    if rawpy is None:
        print("Validation skipped: rawpy is not installed.")
        return

    with rawpy.imread(str(output_path)) as raw:
        rgb_preview = raw.postprocess(no_auto_bright=True, use_camera_wb=False)
        print(f"Validation OK: rawpy reopened '{output_path.name}'")
        print(f"  raw size   : {raw.sizes.raw_width} x {raw.sizes.raw_height}")
        print(f"  raw pattern: {raw.raw_pattern.tolist()}")
        print(f"  color desc : {raw.color_desc.decode(errors='ignore')}")
        print(f"  white level: {raw.white_level}")
        print(f"  preview    : {rgb_preview.shape[1]} x {rgb_preview.shape[0]}")


def main() -> None:
    args = parse_args()
    pattern = normalize_pattern(args.pattern)
    conversion = normalize_conversion(args.conversion)
    inputs = list_raw_inputs(args.input, args.input_glob)
    total = len(inputs)
    multiple_inputs = total > 1
    wants_roi = args.select_roi or args.roi is not None or args.roi_only

    if wants_roi and multiple_inputs:
        raise ValueError("ROI mode supports a single input file only.")

    print(f"Bayer pattern: {pattern}")
    print(f"Data conversion: {conversion}")
    print(f"Found {total} RAW file(s).")

    converted = 0
    failures: list[tuple[Path, str]] = []
    for index, input_path in enumerate(inputs, start=1):
        print(f"[{index}/{total}] Processing: {input_path.name}")
        try:
            raw_bytes = np.fromfile(input_path, dtype=np.uint8)
            if raw_bytes.size == 0:
                raise ValueError(f"Input RAW file is empty: {input_path}")

            width, height = infer_dimensions_for_input(
                raw_bytes=raw_bytes,
                conversion=conversion,
                width_arg=args.width,
                height_arg=args.height,
            )
            print(f"  Dimensions: {width} x {height}")

            if wants_roi:
                image, trailing_bytes = decode_raw_plane(raw_bytes, width, height, conversion)
                if conversion == "LINE" and trailing_bytes:
                    print(f"  ROI analysis: ignored {trailing_bytes} trailing padding bytes.")

                if args.roi is not None:
                    roi = parse_roi_spec(args.roi)
                else:
                    roi = select_roi_interactive(image)

                roi = clip_roi(roi, width, height)
                stats = roi_stats(image, roi)
                roi_output = args.roi_output or default_roi_output_path(input_path)
                save_roi(
                    output_path=roi_output,
                    input_path=input_path,
                    width=width,
                    height=height,
                    conversion=conversion,
                    pattern=pattern,
                    stats=stats,
                )
                print(
                    "  ROI selected: "
                    f"x={stats['x']}, y={stats['y']}, w={stats['w']}, h={stats['h']}; "
                    f"mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
                    f"min={stats['min']:.1f}, max={stats['max']:.1f}"
                )
                print(f"  ROI JSON saved: {roi_output}")

            if args.roi_only:
                converted += 1
                continue

            capture_metadata = collect_capture_metadata(
                input_path=input_path,
                default_make=args.make,
                default_model=args.model,
                exposure_unit=args.exposure_unit,
            )
            print(f"  Metadata: {format_metadata_summary(capture_metadata)}")

            output_path = resolve_output_path(
                input_path=input_path,
                output_target=args.output,
                multiple_inputs=multiple_inputs,
            )
            write_dng(
                input_path=input_path,
                output_path=output_path,
                width=width,
                height=height,
                conversion=conversion,
                pattern=pattern,
                black_level=args.black_level,
                white_level=args.white_level,
                make=args.make,
                model=args.model,
                capture_metadata=capture_metadata,
            )
            print(f"  Wrote DNG: {output_path}")
            converted += 1

            if not args.skip_validate:
                validate_dng(output_path)
        except Exception as exc:
            failures.append((input_path, str(exc)))
            print(f"  Failed: {exc}")

    if args.roi_only:
        print(f"ROI-only mode complete: {converted}/{total} file(s) processed.")
    else:
        print(f"Conversion complete: {converted}/{total} file(s) converted.")

    if failures:
        failure_lines = "\n".join(f"- {path.name}: {message}" for path, message in failures)
        raise RuntimeError(f"Some files failed:\n{failure_lines}")


if __name__ == "__main__":
    main()
