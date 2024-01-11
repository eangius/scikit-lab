#!usr/bin/env python

# Internal libraries
from scikitlab.samplers import ScikitSampler

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
