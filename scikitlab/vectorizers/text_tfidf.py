#!usr/bin/env python

# External libraries
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from typing import Callable
from overrides import overrides


# ABOUT: composite scikit component that manages ngram weighted frequency vectors
# like tf-idf or count vectorizers. Since smaller ngrams are more frequent but less
# insightful than larger ngrams, this component allows fair weighting & filtering
# ngrams proportionally to their token size throughout the corpus.
class WeightedNgramVectorizer(FeatureUnion):

    # construct self as concatenation of ngram size transformers.
    def __init__(
        self,
        vectorizer_type: str,                       # Type of the vectorizer "tfidf" or "count"
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
