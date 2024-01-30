#!usr/bin/env python

# Internal libraries
from scikitlab.samplers import ScikitSampler
from scikitlab.vectorizers.encoder import EnumeratedEncoder

# External libraries
import warnings
import pandas as pd
import numpy as np
from functools import reduce
from typing import Callable, Optional, List, Tuple
from overrides import overrides
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.over_sampling import BorderlineSMOTE, RandomOverSampler


class StrataBalancer(ScikitSampler):
    """
    Enforce fairness by balancing a dataset based on a sub-population
    strata of the input. Each strata is defined from a specific set of
    variables in `X` having identical values. This component will modify
    the dataset only at fit time by either over or under sampling for
    equal volumes per strata but has no effect at predict time.
    """

    def __init__(
        self,
        sampling_mode: str,
        columns: list,
        random_state: int = 0,
        **kwargs,
    ):
        """
        :param sampling_mode:  either `over` or `under` sampling
        :param columns: indices or column names in in `X` that define the strata.
        :param random_state: for determinism
        """
        super().__init__(**kwargs)
        self.sampling_mode = sampling_mode
        self.columns = columns
        self.random_state = random_state
        return

    @overrides
    def _fit_resample(self, X, y=None):
        # TODO: test use-cases:
        # 1)   X, y only as np
        # 2.a) X, y only as pd & cols by name
        # 2.b) X, y only as pd & cols by indx
        # 3.a) X, y of mixed np/pd types & cols by name
        # 3.b) X, y of mixed np/pd types & cols by idx
        # 4)   X only as np
        # 5)   X only as pd & cols by name
        # 6)   X only as pd & cols by idx

        # initialize sampling strategy
        if self.sampling_mode.lower() == "over":
            sampler = RandomOverSampler(
                sampling_strategy="not majority",
                random_state=self.random_state,
            )
        elif self.sampling_mode.lower() == "under":
            sampler = RandomUnderSampler(
                sampling_strategy="not minority",
                random_state=self.random_state,
            )
        else:
            warnings.warn(
                f"Unrecognized '{self.sampling_mode}' sampling mode "
                f"in {self.__class__.__name__} will not have any effect."
            )
            return X, y

        # convert inputs to arrays & columns to indices for common
        # processing
        X, orig_type_X, orig_cols_X = self._convert(X)
        y, orig_type_y, orig_cols_y = self._convert(y)
        # TODO: X & y pd/np decorator

        strata_column_idxs = self.columns
        if all(str(col) for col in self.columns):
            if issubclass(orig_type_X, pd.DataFrame):
                strata_column_idxs = [
                    i
                    for i, (col_name, col_type) in enumerate(orig_cols_X)
                    if col_name in self.columns
                ]
            elif issubclass(orig_type_X, pd.Series):
                strata_column_idxs = [0]

        # isolate target strata & balance it. Note that samplers don't accept
        # multi-label y output so we temporary encode them into 1d arrays.
        X_resample, y_resample = self._slice(X, y, strata_column_idxs)
        encoder = EnumeratedEncoder()
        y_resample = encoder.fit_transform(y_resample)
        if encoder.n_classes <= 1:
            return X, y  # already balanced

        X_resample, y_resample = sampler.fit_resample(
            X=X_resample,
            y=y_resample,
        )
        y_resample = encoder.inverse_transform(y_resample)

        # reconstruct to original columns
        target_column_idxs = list(
            range(
                X_resample.shape[1] - (y.shape[1] if y is not None else 0),
                X_resample.shape[1],
            )
        )
        X_resample, y_resample = self._slice(
            X_resample, y_resample, columns=target_column_idxs
        )

        # convert back to original types
        X_resample, *_ = self._convert(
            X_resample, orig_type_X, orig_cols_X, strata_column_idxs
        )
        y_resample, *_ = self._convert(y_resample, orig_type_y, orig_cols_y)
        return X_resample, y_resample

    @staticmethod
    def _slice(X, Y, columns: list) -> tuple:
        """Appends Y into X & slices out columns of X as new Y"""
        Y_new = X[:, columns]
        X_new = np.delete(X, columns, axis=1)
        if Y is not None:
            X_new = np.concatenate((X_new, Y), axis=1)
            # <<dbg TODO: fix order change via np.insert()
        return X_new, Y_new

    @staticmethod
    def _convert(
        Z,
        collection_type=None,
        column_types: Optional[List[Tuple]] = None,
        slice_idx: Optional[List[int]] = None,
    ) -> tuple:
        """Converts input arrays to & from numpy"""
        if collection_type:  # restore to orig
            if collection_type is pd.DataFrame:
                # since _slice() changes data column order to the end, this
                # ensures original col_names of strata are also reflected.
                lst = column_types.copy()
                column_types = lst + (
                    list(
                        reversed(
                            [lst.pop(idx) for idx in sorted(slice_idx, reverse=True)]
                        )
                    )
                    if slice_idx is not None
                    else []
                )
                Z = pd.DataFrame(Z, columns=[name for name, _ in column_types])
            if collection_type is pd.Series:
                Z = pd.Series(Z, name=column_types[0])

            for name, tpe in column_types:
                Z[name] = Z[name].astype(tpe)

            return Z, collection_type
        else:  # convert to numpy
            collection_type = type(Z)
            if issubclass(collection_type, pd.DataFrame):
                column_types = list(Z.dtypes.items())
                Z = Z.to_numpy()
            elif issubclass(collection_type, pd.Series):
                column_types = [(Z.name, Z.dtypes)]
                Z = Z.to_numpy()
            else:
                column_types = []

        return Z, collection_type, column_types


class RegressionBalancer(ScikitSampler):
    """
    Over or under samples a regression dataset based on a category mapping
    over the target variables. This is useful when certain ranges in the
    predict regress variable are rare.
    """

    def __init__(
        self,
        sampling_mode: str,
        fn_classifier: Callable = None,  # no classification
        random_state: int = 0,
        **kwargs,
    ):
        """
        :param sampling_mode: either `over` or `under` sampling
        :param fn_classifier: how to triage regression target into class ranges
        :param random_state:  for determinism
        """
        super().__init__(**kwargs)
        self.sampling_mode = sampling_mode
        self.fn_classifier = fn_classifier
        self.random_state = random_state
        return

    @overrides
    def _fit_resample(self, X, y):
        # classify output into a temporary target
        orig_type = type(y)
        y = (
            y.to_numpy()
            if isinstance(y, (pd.DataFrame, pd.Series))
            else np.array(y)
            if isinstance(y, list)
            else y
        ).ravel()
        y_cls = pd.Series(y).apply(self.fn_classifier).astype("category")
        sampling_dist = y_cls.value_counts().to_dict()

        # clone minority classes to match the majority
        if self.sampling_mode.lower() == "over":
            target_cls, target_n = reduce(
                lambda item_acc, item_x: item_acc
                if item_acc[1] > item_x[1]
                else item_x,
                sampling_dist.items(),
            )
            sampler = RandomOverSampler(
                sampling_strategy={
                    cls: target_n for cls in sampling_dist if cls != target_cls
                },
                random_state=self.random_state,
            )

        # delete from majority classes to match the minority
        elif self.sampling_mode.lower() == "under":
            target_cls, target_n = reduce(
                lambda item_acc, item_x: item_acc
                if item_acc[1] < item_x[1]
                else item_x,
                sampling_dist.items(),
            )
            sampler = RandomUnderSampler(
                sampling_strategy={
                    cls: target_n for cls in sampling_dist if cls != target_cls
                },
                random_state=self.random_state,
            )

        # do nothing
        else:
            warnings.warn(
                f"Unrecognized '{self.sampling_mode}' sampling mode "
                f"in {self.__class__.__name__} will not have any effect."
            )
            return X, y

        # resample & align indices of classified outputs back to original
        X_resample, y_cls = sampler.fit_resample(X, y_cls.to_numpy())
        y_resample = y[sampler.sample_indices_]
        y_resample = (
            pd.DataFrame(y_resample)
            if orig_type is pd.DataFrame
            else pd.Series(y_resample)
            if orig_type is pd.Series
            else y_resample
        )
        return X_resample, y_resample


class VectorBalancer(ScikitSampler):
    """
    Balance training data for proper learning. This components can over-sample
    the non-majority classes near the decision boundary to generate synthetic
    but relevant examples as well under-sample from all classes near the decision
    boundary to cleanup for noisy data. In order to interpolate the space, this
    component operates on vectors rather than raw datapoints.
    """

    def __init__(
        self,
        deduplicate: bool = False,
        down_sample: bool = False,
        over_sample: bool = False,
        synthesize: bool = False,
        random_state: int = 0,
        **kwargs,
    ):
        """
        :param deduplicate:  remove duplicate / identical vectors
        :param down_sample:  remove redundant vectors
        :param over_sample:  clone minority vectors
        :param synthesize:   generate borderline synthetic vectors
        :param random_state: for determinism
        """
        super().__init__(**kwargs)
        self.deduplicate = deduplicate
        self.down_sample = down_sample
        self.over_sample = over_sample
        self.synthesize = synthesize
        self.random_state = random_state
        return

    @overrides
    def _fit_resample(self, X, y):
        vtrs_x, vtrs_y = X, y

        # de-emphasize frequent by removing redundant label/vector duplicates.
        if self.deduplicate:
            df = pd.DataFrame(data=vtrs_x)
            df["lbl"] = vtrs_y
            df = df.drop_duplicates(keep="first")
            cols = len(df.columns)
            vtrs_x = df.iloc[:, : (cols - 1)].values
            vtrs_y = df["lbl"].values

        # NOTE: following under/over sampling does not apply when vectors encode
        # text sequence because results in invalid out-of-domain dimensions in
        # downstream components.

        # under-sample by cleaning inconsistent
        if self.down_sample:
            vtrs_x, vtrs_y = list(
                zip(
                    *EditedNearestNeighbours(
                        n_neighbors=11,  # lower is too aggressive
                        kind_sel="mode",  # remove when majority disagrees with label
                        sampling_strategy="majority",
                    ).fit_resample(X=vtrs_x, y=vtrs_y)
                )
            )

        # over-sample by injecting synthetic
        if self.synthesize:
            vtrs_x, vtrs_y = list(
                zip(
                    *BorderlineSMOTE(
                        k_neighbors=5,
                        m_neighbors=10,
                        sampling_strategy="not majority",
                        random_state=self.random_state,
                    ).fit_resample(X=vtrs_x, y=vtrs_y)
                )
            )

        # over-sample any remaining imbalance by randomly duping minority cases
        if self.over_sample:
            vtrs_x, vtrs_y = list(
                zip(
                    *RandomOverSampler(
                        sampling_strategy="not majority",
                        random_state=self.random_state,
                    ).fit_resample(X=vtrs_x, y=vtrs_y)
                )
            )

        return vtrs_x, vtrs_y
