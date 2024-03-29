# Vectorizers

These are pipeline components responsible for transforming input objects into
their vectorized form. The input objects are domain specific but the resulting
vectors are the numerical representation suitable for machine learning. Think
of vectorizers as signal generators that emphasize or suppress the aspects being
learned & come in the following forms:

| Type     | Description                            |
|----------|----------------------------------------|
| encoders | transform inputs into vectorized forms |
| reducers | dimensionality reduce other encodings  |

---
::: scikitlab.vectorizers.encoder.EnumeratedEncoder
---
::: scikitlab.vectorizers.frequential.ItemCountVectorizer
---
::: scikitlab.vectorizers.spatial.GeoVectorizer
---
::: scikitlab.vectorizers.temporal.DateTimeVectorizer
---
::: scikitlab.vectorizers.temporal.PeriodicityVectorizer
---
::: scikitlab.vectorizers.text.WeightedNgramVectorizer
---
::: scikitlab.vectorizers.text.UniversalSentenceEncoder
