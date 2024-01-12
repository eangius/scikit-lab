# Samplers

These are pipeline components responsible for changing training data volumes
to match desired distributions to learn from. Useful when data is imbalanced
or when it is scarce/expensive to obtain. Samplers typically implement filtering,
cloning, synthesizing or permuting strategies that modify input rows only at
learning time but have NO effect when predicting.

Integrating samplers within model pipeline ensures downstream data assumptions are
preserved & allows for experimentation to treating data distributions & volumes
as hyper-tunable parameters.

---
::: scikitlab.samplers.balancing.RegressionBalancer
---
::: scikitlab.samplers.balancing.VectorBalancer
