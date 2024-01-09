# Normalizers

These are pipeline components that transform the input format or content while
preserving their original domain semantics. Think of normalizers as necessary
pre-processors that standardize data for downstream steps. Some types of normalizers
include:

| Type       | Description                                                                            |
|------------|----------------------------------------------------------------------------------------|
| scalers    | adjust values within a target range <br> ie: ``MinMaxScaler``, ``ImageCropper``        |
| converters | modify values to a common range <br> ie: ``LanguageTranslator``, ``CurrencyConverter`` |
| formatters | change the structure of the object <br> ie: ``PDFConverter``                           |
| enrichers  | add context or details to the objects <br> ie: ``NameEntityTagger``, ``Q&ALLM``        |


---
::: scikitlab.normalizers.sparsity.DenseTransformer
---
::: scikitlab.normalizers.sparsity.SparseTransformer
