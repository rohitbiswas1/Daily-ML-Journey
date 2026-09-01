# Day 6 — Probability Basics

## What I learned
Probability gives us a way to represent uncertainty. It is useful in ML because predictions are often uncertain rather than simply true or false.

### Core ideas
- **Experiment:** a process with an uncertain outcome.
- **Outcome:** one possible result of an experiment.
- **Event:** a set of outcomes we care about.
- **Probability:** a value from 0 to 1 representing how likely an event is.
- **Complement:** `P(not A) = 1 - P(A)`.

For equally likely outcomes:

`P(A) = favorable outcomes / total outcomes`

### Example
For a fair six-sided die:
- `P(even) = 3/6 = 0.5`
- `P(6) = 1/6 ≈ 0.167`
- `P(greater than 4) = 2/6 ≈ 0.333`

## Empirical probability
We can estimate probability from repeated observations:

`empirical probability = event count / number of trials`

With enough independent trials, empirical results generally become closer to the theoretical probability. Random variation still exists.

## ML intuition
- Classification models can output probabilities such as the estimated probability that an observation belongs to a class.
- Probability helps quantify uncertainty instead of hiding it behind a single hard label.
- A probability is not automatically a guarantee; model calibration and data quality matter.

## Practice
Run `python_ml/06_probability_basics.py` to simulate 10,000 die rolls, compare empirical and theoretical probabilities, and practice the complement rule.

## Next
Learn distributions and sampling, including why the shape of data matters for ML.
