#!usr/bin/env python

# Internal libraries
from scikitlab.vectorizers import ScikitVectorizer


# External libraries
import array
import numbers
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from operator import itemgetter
from overrides import overrides
from typing import Callable


class ItemCountVectorizer(ScikitVectorizer):
    """
    A general purpose counting vectorizer of arbitrary collection of items
    throughout a dataset. This class was adapted from scikits CountVectorizer
    to generalize items not only to text documents. Class also decouples any
    internal item pre-processing into a user definable normalization function.
    """

    def __init__(
        self,
        fn_norm: Callable = None,
        min_freq: float = 0.0,
        max_freq: float = 1.0,
        max_items: int = None,
        out_of_vocab: str = None,
        binary: bool = False,
        **kwargs,
    ):
        """
        :param fn_norm: how to transform individual items per input.
        :param min_freq: filter rare items bellow this threshold.
        :param max_freq: filter frequent items above this threshold.
        :param max_items: keep only top most frequent items in corpus.
        :param out_of_vocab: feature name for catching un-recognized/filtered items.
        :param binary: simply flag all non-zero counts items.
        """
        super().__init__(**kwargs)
        self.fn_norm = fn_norm
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.max_items = max_items
        self.out_of_vocab = out_of_vocab
        self.binary = binary

        # internal state
        self.vocabulary_ = None  # dict[item -> index]
        self.removed_items_ = None
        return

    @overrides
    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    @overrides
    def fit_transform(self, X, y=None):
        vocabulary, X = self._count_vocab(X, fixed_vocab=False)
        if self.binary:
            X.data.fill(1)

        n = X.shape[0]
        max_doc_count = (
            self.max_freq
            if isinstance(self.max_freq, numbers.Integral)
            else self.max_freq * n
        )
        min_doc_count = (
            self.min_freq
            if isinstance(self.min_freq, numbers.Integral)
            else self.min_freq * n
        )
        if max_doc_count < min_doc_count:
            raise ValueError("max_freq corresponds to < documents than min_freq")

        if self.max_items is not None:
            X = self._sort_features(X, vocabulary)

        X, self.removed_items_ = self._limit_features(
            X, vocabulary, max_doc_count, min_doc_count, self.max_items
        )
        if self.max_items is None:
            X = self._sort_features(X, vocabulary)

        self.vocabulary_ = vocabulary
        return X

    @overrides
    def transform(self, X, y=None):
        _, X = self._count_vocab(X, fixed_vocab=True)
        if self.binary:
            X.data.fill(1)
        return X

    # needed for downstream components (ie: TfidfTransformer) to preserve feature names
    def inverse_transform(self, X):
        """
        Return terms per document with nonzero entries in `X`.
        :param X: {array-like, sparse matrix} of shape (n_samples, n_features) document term matrix.
        :return: list of arrays of shape (n_samples,)
        """
        # We need CSR format for fast row manipulations.
        n_samples = X.shape[0]

        terms = np.array(list(self.vocabulary_.keys()))
        indices = np.array(list(self.vocabulary_.values()))
        inverse_vocabulary = terms[np.argsort(indices)]

        if sp.issparse(X):
            return [
                inverse_vocabulary[X[i, :].nonzero()[1]].ravel()
                for i in range(n_samples)
            ]
        else:
            return [
                inverse_vocabulary[np.flatnonzero(X[i, :])].ravel()
                for i in range(n_samples)
            ]

    # get names of the items
    def get_feature_names_out(self, _unused_input_features=None):
        return np.asarray(
            [t for t, i in sorted(self.vocabulary_.items(), key=itemgetter(1))],
            dtype=object,
        )

    def _count_vocab(self, raw_documents, fixed_vocab: bool = False) -> tuple:
        """Create sparse feature matrix, and vocabulary where fixed_vocab=False"""
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            # Add a new value when a new vocabulary item is seen
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__
            if self.out_of_vocab:
                vocabulary[self.out_of_vocab] = 0

        j_indices, indptr = [], []

        values = array.array("i")
        indptr.append(0)
        for doc in raw_documents:
            feature_counter = {}
            for feature in doc:
                feature = self.fn_norm(feature) if self.fn_norm else feature
                try:
                    feature_idx = vocabulary[feature]
                    if feature_idx not in feature_counter:
                        feature_counter[feature_idx] = 1
                    else:
                        feature_counter[feature_idx] += 1
                except KeyError:
                    if self.out_of_vocab:
                        vocabulary[self.out_of_vocab] = 1
                    continue  # ignore out-of-vocab items

            j_indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            indptr.append(len(j_indices))

        if not fixed_vocab:  # disable defaultdict behaviour
            vocabulary = dict(vocabulary)

        indices_dtype = np.int64 if indptr[-1] > np.iinfo(np.int32).max else np.int32
        j_indices = np.asarray(j_indices, dtype=indices_dtype)
        indptr = np.asarray(indptr, dtype=indices_dtype)
        values = np.frombuffer(values, dtype=np.intc)

        X = sp.csr_matrix(
            (values, j_indices, indptr),
            shape=(len(indptr) - 1, len(vocabulary)),
            dtype=np.int64,
        )
        X.sort_indices()
        return vocabulary, X

    @staticmethod
    def _limit_features(X, vocabulary, high=None, low=None, limit=None) -> tuple:
        """Remove too rare or too common features.
        Prune features that are non zero in more samples than high or less
        documents than low, modifying the vocabulary, and restricting it to
        at most the limit most frequent.
        This does not prune samples with zero features.
        """
        removed_terms = set()
        if high is None and low is None and limit is None:
            return X, removed_terms

        # Calculate a mask based on document frequencies
        dfs = (
            np.bincount(X.indices, minlength=X.shape[1])
            if sp.isspmatrix_csr(X)
            else np.diff(X.indptr)
        )

        # TODO: avoid filtering vocabulary[0] if self.out_of_vocab is set.
        mask = np.ones(len(dfs), dtype=bool)
        if high is not None:
            mask &= dfs <= high
        if low is not None:
            mask &= dfs >= low
        if limit is not None and mask.sum() > limit:
            tfs = np.asarray(X.sum(axis=0)).ravel()
            mask_inds = (-tfs[mask]).argsort()[:limit]
            new_mask = np.zeros(len(dfs), dtype=bool)
            new_mask[np.where(mask)[0][mask_inds]] = True
            mask = new_mask

        new_indices = np.cumsum(mask) - 1  # maps old indices to new
        for term, old_index in list(vocabulary.items()):
            if mask[old_index]:
                vocabulary[term] = new_indices[old_index]
            else:
                del vocabulary[term]
                removed_terms.add(term)
        kept_indices = np.where(mask)[0]
        return X[:, kept_indices], removed_terms

    @staticmethod
    def _sort_features(X, vocabulary):
        """Sort features by name
        Returns a reordered matrix and modifies the vocabulary in place
        """
        sorted_features = sorted(vocabulary.items())
        map_index = np.empty(len(sorted_features), dtype=X.indices.dtype)
        for new_val, (term, old_val) in enumerate(sorted_features):
            vocabulary[term] = new_val
            map_index[old_val] = new_val

        X.indices = map_index.take(X.indices, mode="clip")
        return X
