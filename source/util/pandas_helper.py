"""Define helper functions for consistency accross code."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd


def to_csv(
    df: pd.DataFrame,
    path: Path | None = None,
    *,
    sep: str = ",",
    **kwargs: Any,
) -> None:
    """Save the given dataframe as csv."""
    if path is None:
        return
    df.to_csv(path_or_buf=path, sep=sep, **kwargs)
    logging.info(f"Saved df to {path}")
