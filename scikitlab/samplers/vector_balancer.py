#!usr/bin/env python

# Internal libraries
from scikitlab.samplers import ScikitSampler
from scikitlab.data_types import DataSet

# External libraries
import pandas as pd
from overrides import overrides
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import RandomOverSampler


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
        deduplicate: bool = False,  # remove duplicate vectors
        down_sample: bool = False,  # remove redundant vectors
        over_sample: bool = False,  # clone minority vectors
        synthesize: bool = False,  # generate borderline synthetic vectors
        random_state: int = 0,  # determinism
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.deduplicate = deduplicate
        self.down_sample = down_sample
        self.over_sample = over_sample
        self.synthesize = synthesize
        self.random_state = random_state
        return

    @overrides
    def _fit_resample(self, X, y):
        vtrs = DataSet(x=X, y=y)

        # de-emphasize frequent by removing redundant label/vector duplicates.
        if self.deduplicate:
            df = pd.DataFrame(data=vtrs.x)
            df["lbl"] = vtrs.y
            df = df.drop_duplicates(keep="first")
            cols = len(df.columns)
            vtrs = DataSet(x=df.iloc[:, : (cols - 1)].values, y=df["lbl"].values)

        # NOTE: following under/over sampling does not apply when vectors encode
        # text sequence because results in invalid out-of-domain dimensions in
        # downstream components.

        # under-sample by cleaning inconsistent
        if self.down_sample:
            vtrs = DataSet.from_tuple(
                EditedNearestNeighbours(
                    n_neighbors=11,  # lower is too aggressive
                    kind_sel="mode",  # remove when majority disagrees with label
                    sampling_strategy="majority",
                ).fit_resample(
                    X=vtrs.x,
                    y=vtrs.y,
                )
            )

        # over-sample by injecting synthetic
        if self.synthesize:
            vtrs = DataSet.from_tuple(
                BorderlineSMOTE(
                    k_neighbors=5,
                    m_neighbors=10,
                    sampling_strategy="not majority",
                    random_state=self.random_state,
                ).fit_resample(
                    X=vtrs.x,
                    y=vtrs.y,
                )
            )

        # over-sample any remaining imbalance by randomly duping minority cases
        if self.over_sample:
            vtrs = DataSet.from_tuple(
                RandomOverSampler(
                    sampling_strategy="not majority",
                    random_state=self.random_state,
                ).fit_resample(X=vtrs.x, y=vtrs.y)
            )

        return vtrs.x, vtrs.y
