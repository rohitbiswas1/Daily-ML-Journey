# Day 8 — Correlation and Covariance

**Date:** 2026-09-03

## What I learned
- Covariance describes the direction in which two variables tend to move together.
- Pearson correlation standardizes this relationship to a range from -1 to +1.
- Positive correlation means variables tend to increase together; negative correlation means they tend to move in opposite directions.
- Correlation is unitless, while covariance depends on the units of the variables.
- Correlation shows association, not cause and effect, and does not capture every nonlinear relationship.

## What I practiced
- Loaded the study-progress CSV with Pandas.
- Calculated a correlation matrix for study hours, assignments completed, and score.
- Calculated the covariance matrix.
- Verified the study-hours/score relationship with NumPy.
- Interpreted correlation from an ML feature-analysis perspective.

## Reflection
Correlation is a useful first diagnostic when exploring features, but it should not be treated as proof that one variable causes another. I also learned why correlation is more convenient than covariance when comparing relationships measured in different units.

## Next
Learn data visualization with scatter plots and distributions to identify patterns, relationships, and unusual observations.
