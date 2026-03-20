from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import tifffile

try:
    import rawpy
except ImportError:
    rawpy = None


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an 8-bit Bayer RAW buffer into a minimal uncompressed DNG."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("1.raw"),
        help="Input RAW file path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("1.dng"),
        help="Output DNG file path.",
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
        "--skip-validate",
        action="store_true",
        help="Skip rawpy re-open validation after writing the DNG.",
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
) -> list[tuple[int, tifffile.DATATYPE, int, object, bool]]:
    black_levels = (black_level, black_level, black_level, black_level)
    black_level_rational = []
    for value in black_levels:
        black_level_rational.extend((int(value), 1))

    return [
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
) -> None:
    raw_bytes = np.fromfile(input_path, dtype=np.uint8)
    image, trailing_bytes = decode_raw_plane(
        raw_bytes=raw_bytes,
        width=width,
        height=height,
        conversion=conversion,
    )
    if conversion == "LINE" and trailing_bytes:
        print(f"LINE conversion: ignored {trailing_bytes} trailing padding bytes.")
    unique_camera_model = f"{model}-{pattern}-{width}x{height}"
    extratags = build_extratags(
        pattern=pattern,
        black_level=black_level,
        white_level=white_level,
        unique_camera_model=unique_camera_model,
        make=make,
        model=model,
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

    raw_bytes = np.fromfile(args.input, dtype=np.uint8)
    if raw_bytes.size == 0:
        raise ValueError(f"Input RAW file is empty: {args.input}")

    if (args.width is None) ^ (args.height is None):
        raise ValueError("Please provide both --width and --height, or neither.")

    if args.width is not None and args.height is not None:
        width, height = args.width, args.height
        print(f"Using explicit dimensions: {width} x {height}")
    elif conversion == "LINE":
        width, height = 2592, 1944
        print(f"Using LINE defaults: {width} x {height}")
    else:
        width, height = infer_dimensions(raw_bytes)
        print(f"Inferred dimensions from RAW byte layout: {width} x {height}")

    print(f"Bayer pattern: {pattern}")
    print(f"Data conversion: {conversion}")
    write_dng(
        input_path=args.input,
        output_path=args.output,
        width=width,
        height=height,
        conversion=conversion,
        pattern=pattern,
        black_level=args.black_level,
        white_level=args.white_level,
        make=args.make,
        model=args.model,
    )
    print(f"Wrote DNG: {args.output}")

    if not args.skip_validate:
        validate_dng(args.output)


if __name__ == "__main__":
    main()
