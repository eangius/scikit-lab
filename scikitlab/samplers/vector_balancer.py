#!usr/bin/env python

# Internal libraries
from ml.lib.components.samplers import ScikitSampler
from ml.lib.common.data_types import DataSet
from utilities.logger import setup_logger

# External libraries
import pandas as pd
from typing import Set, Union
from overrides import overrides
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import RandomOverSampler


# Settings
logger = setup_logger(__name__.split('.', 1)[0])


# ABOUT: Balance training dataset for proper learning. This over-samples non-majority class
# near the decision boundary to generate synthetic but relevant new examples. Also, under-sample
# from all classes near the decision boundary to cleanup for bad data. This phase is needed after
# the vector rather than the data sampling phase to be able to produce plausible vectors for text.
# Any pre-vector pipeline components are fitted without these.
class VectorBalancer(ScikitSampler):
    def __init__(
        self,
        deduplicate: bool = False,  # remove duplicate vectors
        down_sample: bool = False,  # remove redundant vectors
        over_sample: bool = False,  # clone minority vectors
        synthesize: bool = False,   # generate borderline synthetic vectors
        random_state: int = 0,      # determinism
        **kwargs
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
        size1 = vtrs.x.shape[0]

        logger.debug("balancing vectors from: %s instances", f"{vtrs.x.shape[0]:,}")

        # de-emphasize frequent by removing redundant label/vector duplicates.
        if self.deduplicate:
            df = pd.DataFrame(data=vtrs.x)
            df['lbl'] = vtrs.y
            df.drop_duplicates(
                keep="first",
                inplace=True,
            )
            cols = len(df.columns)
            vtrs = DataSet(
                x=df.iloc[:, :(cols - 1)].values,
                y=df['lbl'].values
            )
            logger.debug("de-duplicated vectors to: %s instances", f"{vtrs.x.shape[0]:,}")

        # NOTE: following under/over sampling does not apply when vectors encode
        # text sequence because results in invalid out-of-domain dimensions in
        # downstream components.

        # under-sample by cleaning inconsistent
        if self.down_sample:
            vtrs = DataSet.from_tuple(EditedNearestNeighbours(
                n_neighbors=11,   # lower is too aggressive
                kind_sel='mode',  # remove when majority disagrees with label
                sampling_strategy='majority',
            ).fit_resample(
                X=vtrs.x,
                y=vtrs.y,
            ))
            logger.debug("under-sampled vectors to: %s instances", f"{vtrs.x.shape[0]:,}")

        # over-sample by injecting synthetic
        if self.synthesize:
            vtrs = DataSet.from_tuple(BorderlineSMOTE(
                k_neighbors=5,
                m_neighbors=10,
                sampling_strategy='not majority',
                random_state=self.random_state,
            ).fit_resample(
                X=vtrs.x,
                y=vtrs.y,
            ))
            logger.debug("synthesized new vectors to: %s instances", f"{vtrs.x.shape[0]:,}")

        # over-sample any remaining imbalance by randomly duping minority cases
        if self.over_sample:
            vtrs = DataSet.from_tuple(RandomOverSampler(
                sampling_strategy='not majority',
                random_state=self.random_state,
            ).fit_resample(
                X=vtrs.x,
                y=vtrs.y
            ))
            logger.debug("cloned vectors to: %s instances", f"{vtrs.x.shape[0]:,}")

        size2 = vtrs.x.shape[0]
        logger.info(
            f"balanced vectors by {100 * (size2 - size1) / size1:2.1f}% "
            f"({size1:,} -> {size2:,} instances)",
        )
        return vtrs.x, vtrs.y

    @property
    def _sampling_type(self):
        return "ensemble"
