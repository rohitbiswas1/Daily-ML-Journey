# Day 6 — Probability Basics

**Date:** 2026-09-01

## What I learned
- Probability represents uncertainty using values from 0 to 1.
- An event is a set of possible outcomes.
- For equally likely outcomes, probability is favorable outcomes divided by total outcomes.
- The complement rule is `P(not A) = 1 - P(A)`.
- Empirical probability estimates likelihood from observed data and can differ from theoretical probability because of random variation.

## What I practiced
- Calculated probabilities for events on a fair six-sided die.
- Simulated 10,000 die rolls with NumPy.
- Compared empirical probabilities with theoretical probabilities.
- Practiced the complement rule.

## Experiment result
The theoretical probability of rolling an even number is **0.5**, while the 10,000-roll simulation gives an empirical value close to 0.5. The small difference demonstrates normal sampling variation.

## Reflection
Probability makes uncertainty measurable. This is important for ML because many models can provide confidence or class probabilities rather than only a hard prediction.

## Next
Learn distributions and sampling, including why the shape of data matters for ML.
