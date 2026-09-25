"""Day 29: Monitoring, data drift, and model reliability."""
import numpy as np

RNG = np.random.default_rng(42)

def psi(reference, current, bins=10):
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    eps = 1e-6
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), eps, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

def main():
    reference = RNG.normal(50.0, 10.0, 1000)
    current = RNG.normal(56.0, 10.0, 1000)
    score = psi(reference, current)
    reference_positive_rate = 0.18
    current_positive_rate = 0.27
    print("Day 29 - Monitoring, Data Drift & Model Reliability")
    print(f"Reference mean: {reference.mean():.2f}")
    print(f"Current mean:   {current.mean():.2f}")
    print(f"PSI:             {score:.4f}")
    print(f"Positive-rate change: {current_positive_rate - reference_positive_rate:+.2%}")
    print("\nMonitor: schema/data quality, feature drift, prediction drift, and fresh-label performance.")
    assert np.isfinite(score)

if __name__ == "__main__":
    main()
