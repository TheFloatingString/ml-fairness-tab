"""Fairness-labeled tabular datasets for the hardware/precision experiment.

Each source dataset is shuffled once (seed 42) and split 60% train / 20%
calibration / 20% test, stratified on the label. The full calibration and
test slices are kept; only the in-context training set fed to the tabular
foundation models (TabPFN / TabICL / TabDPT) is capped at ``CONTEXT_CAP``
rows, since in-context inference does not scale past a few thousand context
rows. XGBoost / MLP train on the full 80% (train + calibration) instead --
see ``X_train_full`` / ``y_train_full``.

Every dataset carries subgroup labels for *all* the sensitive attributes it
exposes -- ``race``, ``gender``, ``age`` -- as columns of a DataFrame, so the
fairness report can break every metric down by each of them.

Sources: UCI (via OpenML) for adult / german_credit / bank_marketing;
folktables (one state-year of ACS PUMS) for acs_income / acs_coverage.
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

SEED = 42
# In-context training-set cap for TabPFN/TabICL/TabDPT. Calibration and test
# slices are never capped; XGBoost/MLP use the uncapped 80% split. TabICL's
# 32-member ensemble runs full O(context^2) attention, so the context is kept
# small to fit a 16 GB T4 (the reference GPU) with headroom.
CONTEXT_CAP = 1000

AGE_BINS = [-np.inf, 25, 45, 60, np.inf]
AGE_LABELS = ["<25", "25-45", "45-60", ">60"]

# ACS PUMS RAC1P recode (2018 1-year).
RAC1P_LABELS = {
    1: "White",
    2: "Black",
    3: "American Indian",
    4: "Alaska Native",
    5: "American Indian / Alaska Native (combos)",
    6: "Asian",
    7: "Native Hawaiian / Pacific Islander",
    8: "Other",
    9: "Two or more races",
}


@dataclass
class FairnessSplit:
    name: str
    # In-context training set (<= CONTEXT_CAP rows).
    X_train: np.ndarray
    y_train: np.ndarray
    # Full 60% training slice, uncapped -- for XGBoost/MLP baselines.
    X_train_full: np.ndarray
    y_train_full: np.ndarray
    X_calib: np.ndarray
    y_calib: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    # One column per sensitive attribute available for this dataset.
    sensitive_train: pd.DataFrame
    sensitive_calib: pd.DataFrame
    sensitive_test: pd.DataFrame


def _age_bands(s: pd.Series) -> pd.Series:
    return pd.cut(
        s.astype(float), bins=AGE_BINS, labels=AGE_LABELS, right=False
    ).astype(str)


def _encode_and_split(
    X: pd.DataFrame,
    y: np.ndarray,
    sensitive: pd.DataFrame,
    name: str,
) -> FairnessSplit:
    cat_cols = X.select_dtypes(include=["category", "object"]).columns
    X = X.copy()
    X[cat_cols] = OrdinalEncoder().fit_transform(X[cat_cols].astype(str))
    X = X.to_numpy(dtype=np.float32)
    sensitive = sensitive.reset_index(drop=True)

    # 60% train / 40% (calib+test), then split the 40% in half -> 20 / 20.
    X_tr, X_tmp, y_tr, y_tmp, s_tr, s_tmp = train_test_split(
        X, y, sensitive, test_size=0.4, random_state=SEED, stratify=y
    )
    X_ca, X_te, y_ca, y_te, s_ca, s_te = train_test_split(
        X_tmp, y_tmp, s_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp
    )

    s_tr = s_tr.reset_index(drop=True)
    s_ca = s_ca.reset_index(drop=True)
    s_te = s_te.reset_index(drop=True)

    # Cap the in-context training set; keep the full slice for the baselines.
    if len(X_tr) > CONTEXT_CAP:
        idx = np.sort(
            np.random.default_rng(SEED).choice(len(X_tr), CONTEXT_CAP, replace=False)
        )
        X_ctx, y_ctx, s_ctx = X_tr[idx], y_tr[idx], s_tr.iloc[idx].reset_index(drop=True)
    else:
        X_ctx, y_ctx, s_ctx = X_tr, y_tr, s_tr

    return FairnessSplit(
        name=name,
        X_train=X_ctx,
        y_train=y_ctx,
        X_train_full=X_tr,
        y_train_full=y_tr,
        X_calib=X_ca,
        y_calib=y_ca,
        X_test=X_te,
        y_test=y_te,
        sensitive_train=s_ctx,
        sensitive_calib=s_ca,
        sensitive_test=s_te,
    )


def load_adult() -> FairnessSplit:
    """UCI Adult income (via OpenML), sensitive attributes = gender (sex),
    race, age band. Positive label = income >50K."""
    bunch = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = bunch.frame.dropna()

    sensitive = pd.DataFrame(
        {
            "gender": df["sex"].astype(str),
            "race": df["race"].astype(str),
            "age": _age_bands(df["age"]),
        },
        index=df.index,
    )
    y = (df["class"].astype(str).str.contains(">50K")).astype(int).to_numpy()
    X = df.drop(columns=["class"])
    return _encode_and_split(X, y, sensitive, "adult")


def load_german_credit() -> FairnessSplit:
    """Statlog German Credit (via OpenML `credit-g`), sensitive attributes =
    gender (from `personal_status`) and age band. Positive label = good
    credit risk. No race attribute in this dataset."""
    bunch = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    df = bunch.frame.dropna()

    sensitive = pd.DataFrame(
        {
            "gender": np.where(
                df["personal_status"].astype(str).str.startswith("male"),
                "male",
                "female",
            ),
            "age": _age_bands(df["age"]),
        },
        index=df.index,
    )
    y = (df["class"].astype(str) == "good").astype(int).to_numpy()
    X = df.drop(columns=["class"])
    return _encode_and_split(X, y, sensitive, "german_credit")


def load_bank_marketing() -> FairnessSplit:
    """Bank Marketing (via OpenML `bank-marketing`). Only an age band is
    recoverable as a sensitive attribute -- OpenML anonymises the columns as
    V1..V16 (V1 = age), so gender / race are not present. Positive label =
    subscribed a term deposit. Class is 1=no / 2=yes.
    """
    bunch = fetch_openml("bank-marketing", version=1, as_frame=True, parser="auto")
    df = bunch.frame.dropna()

    sensitive = pd.DataFrame({"age": _age_bands(df["V1"])}, index=df.index)
    y = (df["Class"].astype(str) == "2").astype(int).to_numpy()
    X = df.drop(columns=["Class"])
    return _encode_and_split(X, y, sensitive, "bank_marketing")


ACS_STATE = "CA"
ACS_YEAR = "2018"
# Raw PUMS download cache; point at a persistent path (e.g. the Modal
# Volume) via FOLKTABLES_ROOT to avoid re-downloading on a cache refresh.
ACS_ROOT = os.environ.get("FOLKTABLES_ROOT", "/tmp/folktables")


def _load_acs(problem, name: str) -> FairnessSplit:
    """Shared loader for folktables ACS problems. Sensitive attributes =
    gender (`SEX`), race (`RAC1P`), age band (`AGEP`) -- all of which are
    also among the problem's own features. Downloads one state-year of ACS
    PUMS on first use."""
    from folktables import ACSDataSource

    source = ACSDataSource(
        survey_year=ACS_YEAR, horizon="1-Year", survey="person", root_dir=ACS_ROOT
    )
    acs_data = source.get_data(states=[ACS_STATE], download=True)
    features, label, _ = problem.df_to_pandas(acs_data)

    sensitive = pd.DataFrame(
        {
            "gender": np.where(features["SEX"].astype(int) == 1, "male", "female"),
            "race": features["RAC1P"]
            .astype(int)
            .map(RAC1P_LABELS)
            .fillna("Unknown")
            .astype(str),
            "age": _age_bands(features["AGEP"]),
        },
        index=features.index,
    )
    y = label.iloc[:, 0].astype(int).to_numpy()
    return _encode_and_split(features, y, sensitive, name)


def load_acs_income() -> FairnessSplit:
    """ACSIncome (folktables): predict personal income > $50k. Positive
    label = income > 50k. Sensitive attributes = gender, race, age."""
    from folktables import ACSIncome

    return _load_acs(ACSIncome, "acs_income")


def load_acs_coverage() -> FairnessSplit:
    """ACSPublicCoverage (folktables): predict public health-insurance
    coverage. Positive label = covered. Sensitive attributes = gender,
    race, age."""
    from folktables import ACSPublicCoverage

    return _load_acs(ACSPublicCoverage, "acs_coverage")


DATASETS = {
    "adult": load_adult,
    "german_credit": load_german_credit,
    "bank_marketing": load_bank_marketing,
    "acs_income": load_acs_income,
    "acs_coverage": load_acs_coverage,
}


def load(name: str) -> FairnessSplit:
    try:
        return DATASETS[name]()
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}") from None
