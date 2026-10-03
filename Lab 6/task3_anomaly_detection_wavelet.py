"""
Task 3: Anomaly Detection in Sensor Data Using Wavelet Transformation
======================================================================
Goal: Monitor sensor data from industrial machines, identify anomalies
      in real-time using wavelet transformation.

Expected output matches the two-panel figure shown in the lab sheet:
  - Top   : Sensor Data vs Denoised Signal
  - Bottom: Residuals with detected anomalies (green dots)
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt


# ── Signal generation ─────────────────────────────────────────────────────────

def generate_sensor_data(n_samples=1000, seed=42):
    """
    Simulate an industrial sensor signal: sinusoidal baseline + noise + spikes.
    """
    rng   = np.random.default_rng(seed)
    t     = np.linspace(0, 10 * np.pi, n_samples)

    # Baseline (slow oscillation)
    base  = np.sin(t) * 0.5 + np.cos(0.3 * t) * 0.3

    # Gaussian noise
    noise = rng.normal(0, 0.5, n_samples)

    # Anomalous spikes at random locations
    n_spikes     = 40
    spike_idx    = rng.choice(n_samples, n_spikes, replace=False)
    spike_values = rng.choice([-1, 1], n_spikes) * rng.uniform(1.5, 3.0, n_spikes)

    sensor = base + noise
    sensor[spike_idx] += spike_values

    return t, sensor, spike_idx


# ── Wavelet denoising ─────────────────────────────────────────────────────────

def wavelet_denoise(signal, wavelet="db4", level=5, mode="soft"):
    """
    Denoise a 1-D signal using wavelet thresholding (VisuShrink / universal threshold).

    Args:
        signal  : 1-D numpy array.
        wavelet : Wavelet family (default 'db4').
        level   : Decomposition level.
        mode    : Thresholding mode – 'soft' or 'hard'.

    Returns:
        denoised : Reconstructed signal after thresholding detail coefficients.
        coeffs   : List of wavelet coefficients (for inspection).
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Universal / VisuShrink threshold  σ * sqrt(2 * log(N))
    detail_coeffs = coeffs[1:]
    sigma         = np.median(np.abs(detail_coeffs[-1])) / 0.6745   # MAD estimate
    threshold     = sigma * np.sqrt(2 * np.log(len(signal)))

    # Apply threshold to all detail sub-bands
    new_coeffs    = [coeffs[0]]   # keep approximation unchanged
    for detail in detail_coeffs:
        new_coeffs.append(pywt.threshold(detail, threshold, mode=mode))

    denoised = pywt.waverec(new_coeffs, wavelet)
    # waverec may produce length N+1 due to padding
    return denoised[: len(signal)], coeffs


# ── Anomaly detection ─────────────────────────────────────────────────────────

def detect_anomalies(sensor, denoised, z_threshold=2.5):
    """
    Detect anomalies as residual samples whose |z-score| exceeds z_threshold.

    Args:
        sensor      : Raw sensor signal.
        denoised    : Denoised (baseline) signal.
        z_threshold : Number of standard deviations for anomaly cutoff.

    Returns:
        residuals      : sensor - denoised
        anomaly_mask   : Boolean array, True where anomaly detected.
        anomaly_indices: Integer indices of anomalies.
    """
    residuals   = sensor - denoised
    mean_r      = np.mean(residuals)
    std_r       = np.std(residuals)

    z_scores       = (residuals - mean_r) / (std_r + 1e-9)
    anomaly_mask   = np.abs(z_scores) > z_threshold
    anomaly_indices = np.where(anomaly_mask)[0]

    return residuals, anomaly_mask, anomaly_indices


# ── Visualisation (matches the expected output) ───────────────────────────────

def plot_results(t, sensor, denoised, residuals, anomaly_indices, anomaly_mask,
                 save_path="outputs/task3_anomaly_detection_output.png"):
    """Reproduce the two-panel figure from the lab sheet."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    x_axis = np.arange(len(sensor))

    # ── Panel 1: Sensor vs Denoised ───────────────────────────────────────────
    ax1.plot(x_axis, sensor,   color="steelblue",  linewidth=0.8,  label="Sensor Data")
    ax1.plot(x_axis, denoised, color="darkorange",  linewidth=1.5,
             linestyle="--", label="Denoised Signal")
    ax1.set_title("Sensor Data and Denoised Signal", fontsize=12)
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_ylim(-3, 3)
    ax1.grid(True, linestyle=":", alpha=0.4)

    # ── Panel 2: Residuals + Anomalies ────────────────────────────────────────
    ax2.plot(x_axis, residuals, color="firebrick", linewidth=0.8, label="Residuals")
    ax2.scatter(
        anomaly_indices, residuals[anomaly_indices],
        color="green", s=50, zorder=5, label="Anomalies",
    )
    ax2.set_title("Residuals and Detected Anomalies", fontsize=12)
    ax2.set_ylabel("Residual")
    ax2.set_xlabel("Sample index")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_ylim(-3, 3)
    ax2.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {save_path}")
    plt.show()


# ── Wavelet coefficient plot (bonus) ─────────────────────────────────────────

def plot_wavelet_coefficients(coeffs, wavelet="db4",
                              save_path="outputs/task3_wavelet_coefficients.png"):
    """Show the multi-resolution wavelet decomposition."""
    n_levels = len(coeffs)
    fig, axes = plt.subplots(n_levels, 1, figsize=(12, 2 * n_levels))
    fig.suptitle(f"Wavelet Decomposition ({wavelet})", fontsize=13, fontweight="bold")

    labels = ["Approximation (cA)"] + [f"Detail level {i}" for i in range(1, n_levels)]
    for ax, coef, lbl in zip(axes, coeffs, labels):
        ax.plot(coef, linewidth=0.8)
        ax.set_ylabel(lbl, fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_xlim(0, len(coef))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Wavelet coefficients saved → {save_path}")
    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Task 3: Anomaly Detection Using Wavelet Transformation")
    print("=" * 60)

    N = 1000
    t, sensor, true_spike_idx = generate_sensor_data(n_samples=N)

    print(f"\nSignal length  : {N}")
    print(f"True anomalies : {len(true_spike_idx)} spikes injected")

    # ── Denoise ───────────────────────────────────────────────────────────────
    denoised, coeffs = wavelet_denoise(sensor, wavelet="db4", level=5, mode="soft")

    # ── Detect anomalies ──────────────────────────────────────────────────────
    residuals, anomaly_mask, anomaly_indices = detect_anomalies(
        sensor, denoised, z_threshold=2.5
    )

    print(f"Detected anomalies: {len(anomaly_indices)}")
    print(f"Anomaly indices (first 10): {anomaly_indices[:10]}")

    # Simple performance estimate
    true_set     = set(true_spike_idx)
    detected_set = set(anomaly_indices)
    tp = len(true_set & detected_set)
    fp = len(detected_set - true_set)
    fn = len(true_set - detected_set)
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    print(f"\nPerformance (proximity ±0 samples):")
    print(f"  Precision : {precision:.3f}")
    print(f"  Recall    : {recall:.3f}")
    print(f"  F1 Score  : {f1:.3f}")

    # ── Plot main output (matches expected figure) ────────────────────────────
    plot_results(t, sensor, denoised, residuals, anomaly_indices, anomaly_mask)

    # ── Plot wavelet decomposition ────────────────────────────────────────────
    plot_wavelet_coefficients(coeffs)


if __name__ == "__main__":
    main()
