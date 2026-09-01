"""Day 6: probability basics for machine learning."""

import numpy as np


def main() -> None:
    # A fair six-sided die: every outcome has probability 1/6.
    outcomes = np.arange(1, 7)
    p_even = np.mean(outcomes % 2 == 0)
    p_greater_than_4 = np.mean(outcomes > 4)

    print("Die outcomes:", outcomes.tolist())
    print("P(even):", round(p_even, 3))
    print("P(greater than 4):", round(p_greater_than_4, 3))

    # Empirical probability from repeated random experiments.
    rng = np.random.default_rng(42)
    rolls = rng.integers(1, 7, size=10_000)
    empirical_even = np.mean(rolls % 2 == 0)
    empirical_six = np.mean(rolls == 6)

    print("\n10,000 simulated rolls")
    print("Empirical P(even):", round(empirical_even, 3))
    print("Empirical P(6):", round(empirical_six, 3))

    # Complement rule: P(not A) = 1 - P(A).
    p_at_least_5 = np.mean(outcomes >= 5)
    print("\nP(not at least 5):", round(1 - p_at_least_5, 3))

    print("\nML intuition: probabilities represent uncertainty and are used in areas such as classification.")


if __name__ == "__main__":
    main()
