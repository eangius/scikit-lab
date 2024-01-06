#!usr/bin/env python

# Internal libraries
from scikitlab.vectorizers import ScikitVectorizer

# External libraries
import os
import numpy as np
import pandas as pd
import tensorflow_hub as hub
from functools import cached_property
from pathlib import Path
from zipfile import ZipFile
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from typing import Callable
from overrides import overrides


class WeightedNgramVectorizer(FeatureUnion):
    """
    Composite scikit component that manages ngram weighted frequency vectors like
    tf-idf or count vectorizers. Since smaller ngrams are more frequent but less
    insightful than larger ngrams, this component allows fair weighting & filtering
    ngrams proportionally to their token size throughout the corpus.
    """

    # construct self as concatenation of ngram size transformers.
    def __init__(
        self,
        vectorizer_type: str = 'tfidf',             # Type of the vectorizer "tfidf" or "count"
        weight_fn: Callable[[int], float] = None,   # function to weight n-grams by size.
        ngram_range: tuple = (1, 1),                # min/max ngram sizes to build.
        n_jobs: int = None,                         # parallel process
        verbose: bool = False,                      # verbose
        **kwargs                                    # args for base vectorizer
    ):
        self.vectorizer_type = vectorizer_type
        self.weight_fn = weight_fn
        self.ngram_range = ngram_range

        trans = TfidfVectorizer if vectorizer_type.lower() == 'tfidf' else CountVectorizer
        transformers = {
            n: trans(
                ngram_range=(n, n),
                **kwargs,  # delegated params
            )
            for n in range(ngram_range[0], ngram_range[1] + 1)
        }

        super().__init__(
            transformer_weights={f"{n}gram": weight_fn(n) for n in transformers} if weight_fn else None,
            transformer_list=[(f"{n}gram", trans) for n, trans in transformers.items()],
            n_jobs=n_jobs,
            verbose=verbose
        )
        return

    # strip prefix "ngram__" transformer name
    @overrides
    def get_feature_names_out(self, input_features=None):
        return np.array([
            ngram.split('gram__', 1)[1]
            for ngram in super().get_feature_names_out(input_features)
        ])


class UniversalSentenceEncoder(ScikitVectorizer):
    """
    Wrapper to pre-trained universal sentence encoder model to convert short
    English texts into fixed size dense semantic vectors without need for text
    preprocessing. This component may require warm start but is useful to
    cluster or compare document similarities.
    """

    def __init__(self, resource_dir: str = None):
        self.resource_dir = resource_dir or 'https://tfhub.dev/google/universal-sentence-encoder/4'

    @overrides
    def transform(self, X, y=None):
        X = X[0].tolist() if isinstance(X, pd.DataFrame) else X
        X = super().transform(X, y)
        return self.embedding(X).numpy()

    @cached_property
    def dimensionality(self) -> int:
        return self.transform(['']).shape[-1]

    def get_feature_names_out(self, _unused_input_features=None):
        return np.array([
            f"{self.__class__.__name__}{i}"
            for i in range(self.dimensionality)
        ])

    # dynamically unpack & load since tensorflow model is not serializable
    @cached_property
    def embedding(self):
        if not os.path.exists(self.resource_dir):
            archive = f'{self.resource_dir}.zip'
            if not os.path.exists(archive):
                raise IOError(f"resource {self.resource_dir} not available")

            with ZipFile(archive, 'r') as zip_ref:
                zip_ref.extractall(Path(self.resource_dir).parent)
        return hub.load(self.resource_dir)
