from __future__ import annotations

from math import lgamma, pi
from typing import Dict, Sequence

import numpy as np

RAW_DIST_CHANNELS = ("R", "G1", "G2", "B")
RGB_DIST_CHANNELS = ("R", "G", "B")

MODEL_CODE_MAP = {
    "none": 0.0,
    "gaussian": 1.0,
    "poisson": 2.0,
    "gamma": 3.0,
}

DIST_FEATURE_KEYS = (
    "best_model_code",
    "best_model_margin",
    "gauss_mu",
    "gauss_sigma",
    "gauss_r2",
    "poisson_lambda",
    "gamma_shape",
    "gamma_scale",
    "fano_factor",
    "skewness",
    "kurtosis",
    "entropy",
    "fft_peak_freq",
    "fft_spectral_centroid",
    "fft_spectral_spread",
    "fft_dc_ratio",
    "mgf_t1",
    "mgf_t2",
    "mgf_t3",
    "mgf_t4",
)

_EPS = 1e-12
_LAPLACE_T_SAMPLES = (0.5, 1.0, 2.0, 4.0)


def build_rgb_channels(raw_channels: np.ndarray) -> np.ndarray:
    """Convert Bayer-split raw channels into an RGB cube."""
    if raw_channels.ndim != 3:
        raise ValueError("raw_channels must be a 3D array")
    if raw_channels.shape[-1] == 3:
        return raw_channels.astype(np.float64, copy=False)
    if raw_channels.shape[-1] < 4:
        raise ValueError("Expected 3 or 4 channels to build RGB distributions")
    return np.stack(
        [
            raw_channels[:, :, 0],
            (raw_channels[:, :, 1] + raw_channels[:, :, 2]) / 2.0,
            raw_channels[:, :, 3],
        ],
        axis=-1,
    ).astype(np.float64, copy=False)


def _empty_channel_features(channel_name: str) -> Dict[str, float]:
    return {f"{channel_name}_{key}": 0.0 for key in DIST_FEATURE_KEYS}


def _suggest_histogram_bins(values: np.ndarray) -> int:
    if values.size <= 1:
        return 16
    q25, q75 = np.percentile(values, [25, 75])
    iqr = float(q75 - q25)
    data_range = float(np.max(values) - np.min(values))
    if iqr > 0 and data_range > 0:
        bin_width = 2.0 * iqr / np.cbrt(values.size)
        if bin_width > 0:
            return int(np.clip(np.ceil(data_range / bin_width), 16, 128))
    return int(np.clip(np.ceil(np.sqrt(values.size)), 16, 128))


def _histogram_probabilities(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if values.size == 0:
        return np.zeros(16, dtype=np.float64), np.arange(16, dtype=np.float64), 1.0
    if np.allclose(values, values[0]):
        counts = np.zeros(16, dtype=np.float64)
        counts[0] = 1.0
        centers = np.linspace(values[0], values[0] + 1.0, 16)
        return counts, centers, 1.0

    bins = _suggest_histogram_bins(values)
    counts, edges = np.histogram(values, bins=bins, density=False)
    probs = counts.astype(np.float64)
    probs /= np.sum(probs) + _EPS
    centers = (edges[:-1] + edges[1:]) / 2.0
    bin_width = float(np.mean(np.diff(edges))) if len(edges) > 1 else 1.0
    return probs, centers, max(bin_width, _EPS)


def _safe_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    return float(1.0 - ss_res / (ss_tot + _EPS))


def _normalize_curve(curve: np.ndarray) -> np.ndarray:
    curve = np.maximum(curve, 0.0)
    total = float(np.sum(curve))
    if total <= _EPS:
        return np.full_like(curve, 1.0 / max(len(curve), 1), dtype=np.float64)
    return curve / total


def _gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(sigma, _EPS)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * pi))


def _poisson_log_likelihood(values: np.ndarray, lam: float) -> float:
    lam = max(lam, _EPS)
    int_values = np.rint(np.clip(values, 0, None)).astype(np.int64)
    lgamma_terms = np.fromiter(
        (lgamma(int(v) + 1) for v in int_values),
        dtype=np.float64,
        count=int_values.size,
    )
    return float(np.sum(int_values * np.log(lam) - lam - lgamma_terms))


def _gamma_log_likelihood(values: np.ndarray, shape: float, scale: float) -> float:
    safe_values = np.clip(values, _EPS, None)
    shape = max(shape, _EPS)
    scale = max(scale, _EPS)
    log_pdf = (
        (shape - 1.0) * np.log(safe_values)
        - safe_values / scale
        - shape * np.log(scale)
        - lgamma(shape)
    )
    return float(np.sum(log_pdf))


def _aic(log_likelihood: float, num_params: int) -> float:
    return float(2 * num_params - 2 * log_likelihood)


def _fit_models(values: np.ndarray) -> Dict[str, float]:
    mu = float(np.mean(values))
    var = float(np.var(values))
    sigma = float(np.sqrt(max(var, _EPS)))

    gaussian_ll = float(-0.5 * values.size * (np.log(2.0 * pi * sigma * sigma) + 1.0))
    gaussian_aic = _aic(gaussian_ll, num_params=2)

    poisson_lambda = max(float(np.mean(np.clip(values, 0, None))), _EPS)
    poisson_aic = _aic(_poisson_log_likelihood(values, poisson_lambda), num_params=1)

    gamma_shape = max((mu * mu) / (var + _EPS), _EPS)
    gamma_scale = max(var / (mu + _EPS), _EPS)
    gamma_aic = _aic(
        _gamma_log_likelihood(values, gamma_shape, gamma_scale),
        num_params=2,
    )

    candidates = {
        "gaussian": gaussian_aic,
        "poisson": poisson_aic,
        "gamma": gamma_aic,
    }
    ordered = sorted(candidates.items(), key=lambda item: item[1])
    best_model = ordered[0][0]
    best_margin = float(ordered[1][1] - ordered[0][1]) if len(ordered) > 1 else 0.0

    return {
        "best_model_code": MODEL_CODE_MAP[best_model],
        "best_model_margin": best_margin,
        "gauss_mu": mu,
        "gauss_sigma": sigma,
        "poisson_lambda": poisson_lambda,
        "gamma_shape": gamma_shape,
        "gamma_scale": gamma_scale,
    }


def _compute_frequency_features(probabilities: np.ndarray) -> Dict[str, float]:
    spectrum = np.abs(np.fft.rfft(probabilities))
    if spectrum.size == 0:
        return {
            "fft_peak_freq": 0.0,
            "fft_spectral_centroid": 0.0,
            "fft_spectral_spread": 0.0,
            "fft_dc_ratio": 0.0,
        }

    freqs = np.fft.rfftfreq(probabilities.size, d=1.0)
    dc_ratio = float(spectrum[0] / (np.sum(spectrum) + _EPS))

    if spectrum.size == 1:
        return {
            "fft_peak_freq": 0.0,
            "fft_spectral_centroid": 0.0,
            "fft_spectral_spread": 0.0,
            "fft_dc_ratio": dc_ratio,
        }

    non_dc = spectrum[1:]
    non_dc_freqs = freqs[1:]
    weight_sum = float(np.sum(non_dc))
    if weight_sum <= _EPS:
        return {
            "fft_peak_freq": 0.0,
            "fft_spectral_centroid": 0.0,
            "fft_spectral_spread": 0.0,
            "fft_dc_ratio": dc_ratio,
        }

    peak_idx = int(np.argmax(non_dc))
    centroid = float(np.sum(non_dc_freqs * non_dc) / weight_sum)
    spread = float(
        np.sqrt(np.sum(((non_dc_freqs - centroid) ** 2) * non_dc) / weight_sum)
    )
    return {
        "fft_peak_freq": float(non_dc_freqs[peak_idx]),
        "fft_spectral_centroid": centroid,
        "fft_spectral_spread": spread,
        "fft_dc_ratio": dc_ratio,
    }


def _compute_shape_features(values: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    mu = float(np.mean(values))
    sigma = float(np.std(values))
    sigma_safe = max(sigma, _EPS)
    centered = values - mu
    skewness = float(np.mean((centered / sigma_safe) ** 3))
    kurtosis = float(np.mean((centered / sigma_safe) ** 4) - 3.0)
    fano_factor = float(np.var(values) / (mu + _EPS))
    entropy = float(-np.sum(probabilities * np.log(probabilities + _EPS)))
    return {
        "fano_factor": fano_factor,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "entropy": entropy,
    }


def _compute_laplace_samples(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {f"mgf_t{idx + 1}": 0.0 for idx in range(len(_LAPLACE_T_SAMPLES))}
    value_min = float(np.min(values))
    value_span = float(np.max(values) - value_min)
    scaled = (values - value_min) / (value_span + _EPS)
    return {
        f"mgf_t{idx + 1}": float(np.mean(np.exp(-t * scaled)))
        for idx, t in enumerate(_LAPLACE_T_SAMPLES)
    }


def _compute_channel_features(values: np.ndarray, channel_name: str) -> Dict[str, float]:
    if values.size == 0:
        return _empty_channel_features(channel_name)

    values = values.astype(np.float64, copy=False)
    probabilities, centers, bin_width = _histogram_probabilities(values)
    model_features = _fit_models(values)

    gauss_curve = _gaussian_pdf(
        centers,
        model_features["gauss_mu"],
        model_features["gauss_sigma"],
    ) * bin_width
    gauss_probs = _normalize_curve(gauss_curve)

    channel_features = {
        **model_features,
        "gauss_r2": _safe_r2(probabilities, gauss_probs),
        **_compute_shape_features(values, probabilities),
        **_compute_frequency_features(probabilities),
        **_compute_laplace_samples(values),
    }
    return {
        f"{channel_name}_{feature_name}": float(feature_value)
        for feature_name, feature_value in channel_features.items()
    }


def compute_channel_distribution_features(
    channel_cube: np.ndarray,
    roi_mask: np.ndarray,
    channel_names: Sequence[str],
) -> Dict[str, float]:
    """Compute distribution-model features for each channel inside the ROI."""
    if channel_cube.ndim != 3:
        raise ValueError("channel_cube must be a 3D array")

    channel_count = min(channel_cube.shape[-1], len(channel_names))
    mask = roi_mask > 0
    features: Dict[str, float] = {}
    for idx in range(channel_count):
        channel_name = channel_names[idx]
        channel_values = channel_cube[:, :, idx][mask]
        features.update(_compute_channel_features(channel_values, channel_name))
    return features


def compute_rgb_distribution_features(
    raw_channels: np.ndarray,
    roi_mask: np.ndarray,
) -> Dict[str, float]:
    """Compute RGB distribution-model features using merged green channel."""
    rgb_channels = build_rgb_channels(raw_channels)
    return compute_channel_distribution_features(rgb_channels, roi_mask, RGB_DIST_CHANNELS)
