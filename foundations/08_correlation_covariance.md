# Day 8 — Correlation and Covariance

## What I learned

Correlation and covariance describe how two numerical variables change together. They are useful during exploratory data analysis and feature selection.

### Covariance
Covariance indicates the direction of joint movement:
- Positive covariance: variables tend to move in the same direction.
- Negative covariance: variables tend to move in opposite directions.
- A value near zero suggests little linear co-movement.

Its magnitude depends on the units of the variables, so covariance is harder to compare across datasets.

### Correlation
Pearson correlation is standardized and ranges from **-1 to +1**:
- `+1`: perfect positive linear relationship.
- `0`: no linear relationship.
- `-1`: perfect negative linear relationship.

Correlation is easier to compare because it is unitless.

## Important ML caution
Correlation measures association, not cause and effect. A high correlation between two features does not prove that one produces changes in the other. A value near zero also does not rule out nonlinear relationships.

## What I practiced
- Loaded the study-progress dataset with Pandas.
- Built a Pearson correlation matrix for study hours, assignments completed, and score.
- Built a covariance matrix.
- Recomputed the study-hours/score relationship with NumPy.
- Interpreted the results from an ML perspective.

## Next
Learn how to visualize relationships with scatter plots and distributions, then use visualization to find patterns and unusual observations.
