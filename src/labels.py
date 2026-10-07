"""Label source. Pluggable: gold labels now, report-derived weak labels later.

A weak-label file (e.g. from LLM report labeling) can be passed as a CSV with
StudyInstanceUID + the 12 target columns holding probabilities in [0, 1].
"""
import numpy as np
import pandas as pd

from common import TARGETS


def get_labels(train_csv, weak_csv=None, weak_weight=0.5):
    """Returns (DataFrame indexed by StudyInstanceUID with TARGETS, weight Series).

    Missing labels are NaN; the loss masks them out.
    """
    df = pd.read_csv(train_csv).set_index("StudyInstanceUID")[TARGETS].astype(float)
    weight = pd.Series(np.where(df.notna().all(axis=1), 1.0, 0.0), index=df.index)
    if weak_csv is not None:
        weak = pd.read_csv(weak_csv).set_index("StudyInstanceUID")[TARGETS].reindex(df.index)
        fill = df.isna() & weak.notna()
        df = df.where(~fill, weak)
        weight = weight.where(weight > 0, np.where(fill.any(axis=1), weak_weight, 0.0))
    return df, weight
