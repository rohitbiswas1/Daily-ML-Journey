"""Day 7: distributions and sampling for machine learning."""

import numpy as np


def describe(name: str, values: np.ndarray) -> None:
    print(f"{name}: n={len(values)}, mean={values.mean():.2f}, std={values.std():.2f}")


def main() -> None:
    rng = np.random.default_rng(42)

    # A normal distribution is common for measurements and model noise.
    population = rng.normal(loc=70, scale=10, size=10_000)
    sample = rng.choice(population, size=100, replace=False)

    describe("Population", population)
    describe("Random sample", sample)
    print("Population mean:", round(population.mean(), 2))
    print("Sample mean:", round(sample.mean(), 2))

    # Sampling repeatedly shows how sample means vary around the population mean.
    sample_means = np.array([
        rng.choice(population, size=100, replace=False).mean()
        for _ in range(500)
    ])
    print("\n500 sample means")
    describe("Sample means", sample_means)
    print("5th percentile:", round(np.percentile(sample_means, 5), 2))
    print("95th percentile:", round(np.percentile(sample_means, 95), 2))

    # Compare a roughly symmetric distribution with a right-skewed one.
    symmetric = rng.normal(50, 8, 5_000)
    skewed = rng.exponential(scale=10, size=5_000)
    print("\nDistribution intuition")
    describe("Symmetric", symmetric)
    describe("Right-skewed", skewed)
    print("Symmetric mean vs median:", round(symmetric.mean(), 2), round(np.median(symmetric), 2))
    print("Skewed mean vs median:", round(skewed.mean(), 2), round(np.median(skewed), 2))

    print("\nML intuition: samples are used to estimate population properties, and distribution shape affects how we interpret data and uncertainty.")


if __name__ == "__main__":
    main()
