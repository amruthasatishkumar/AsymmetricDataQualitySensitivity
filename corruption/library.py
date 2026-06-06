"""Corruption library for AgentDQ.

Each corruption is a pure function (df, severity, seed) -> df' that:
  * never touches the label columns (y_t1, y_t2, y_t3, y_t5) or split / order_id
  * is deterministic given (family, severity, seed)
  * preserves row count UNLESS the family is row-level (e.g., duplicate_rows)
  * preserves column count UNLESS the family is schema-level (schema_drift)

Severity is a fraction in [0, 1] interpreted per-family (see docstrings).

Public API:
    FAMILIES                     -- list of registered family names
    apply(df, family, severity, seed)
    apply_clean(df)              -- identity, for symmetry in the harness
    list_families()
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

LABEL_COLS = {"y_t1", "y_t2", "y_t3"}
NON_FEATURE_COLS = LABEL_COLS | {"order_id", "split"}

MONETARY_COLS = {
    "total_price",
    "total_freight",
    "total_payment_value",
    "max_price",
    "mean_price",
}

TIMEZONE_NUMERIC_COLS = {
    "purchase_hour",
    "days_to_estimated",
    "days_to_approved",
    "days_carrier_to_customer",
    "days_purchase_to_delivery",
    "delivery_vs_estimate_days",
}

PORTUGUESE_CAT_COLS = {"dominant_category"}

TEXT_COLS = {"review_comment_title", "review_comment_message"}


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def _numeric_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def _categorical_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [
        c for c in cols
        if (pd.api.types.is_object_dtype(df[c])
            or isinstance(df[c].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(df[c]))
    ]


# ---------------- families (kept) ----------------

def missing_injection(df: pd.DataFrame, severity: float, seed: int) -> pd.DataFrame:
    """Set `severity` fraction of values in feature columns to NaN.

    Real-world: missing values from optional fields, opt-out, sensor drop-outs.
    """
    out = df.copy()
    feats = _feature_cols(out)
    rng = np.random.default_rng(seed)
    for col in feats:
        mask = rng.random(len(out)) < severity
        if mask.any():
            out.loc[mask, col] = np.nan
    return out


def type_flip(df: pd.DataFrame, severity: float, seed: int) -> pd.DataFrame:
    """Coerce `severity` fraction of numeric values into strings.

    Real-world: ETL misconfig, schema-on-read inconsistencies.
    """
    out = df.copy()
    feats = _feature_cols(out)
    nums = _numeric_cols(out, feats)
    rng = np.random.default_rng(seed)
    for col in nums:
        mask = rng.random(len(out)) < severity
        if not mask.any():
            continue
        new = out[col].astype("object")
        flipped = new[mask].apply(lambda v: f"{v}" if pd.notna(v) else v)
        new.loc[mask] = flipped
        out[col] = new
    return out


def label_noise_categorical(df: pd.DataFrame, severity: float, seed: int) -> pd.DataFrame:
    """For each non-text categorical feature, swap `severity` fraction of values
    to a different same-column level (drawn from the marginal distribution).

    Despite the name, this corrupts FEATURES, not the prediction label.
    Real-world: data-entry errors, free-form fields normalized by a flaky pipeline.
    """
    out = df.copy()
    feats = _feature_cols(out)
    cats = [c for c in _categorical_cols(out, feats) if c not in TEXT_COLS]
    rng = np.random.default_rng(seed)
    for col in cats:
        levels = out[col].dropna().astype(str).unique()
        if len(levels) < 2:
            continue
        mask = rng.random(len(out)) < severity
        if not mask.any():
            continue
        choices = rng.choice(levels, size=int(mask.sum()), replace=True)
        new_vals = choices.copy()
        originals = out.loc[mask, col].astype(str).to_numpy()
        for i, (orig, repl) in enumerate(zip(originals, choices)):
            if repl == orig and len(levels) > 1:
                idx = (np.where(levels == orig)[0][0] + 1) % len(levels)
                new_vals[i] = levels[idx]
        out.loc[mask, col] = new_vals
    return out


def duplicate_rows(df: pd.DataFrame, severity: float, seed: int) -> pd.DataFrame:
    """Append duplicates of `severity` fraction of rows.

    Real-world: idempotency bugs, retry-without-dedup in ingestion pipelines.
    """
    out = df.copy()
    n = len(out)
    if n == 0 or severity <= 0:
        return out
    k = int(round(severity * n))
    if k == 0:
        return out
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=k)
    dup = out.iloc[idx].copy()
    return pd.concat([out, dup], ignore_index=True)


def schema_drift(df: pd.DataFrame, severity: float, seed: int) -> pd.DataFrame:
    """Rename `severity` fraction of feature columns with synthetic-looking names.

    Real-world: upstream schema migrations not propagated to consumers.
    """
    out = df.copy()
    feats = _feature_cols(out)
    rng = np.random.default_rng(seed)
    n_rename = int(round(severity * len(feats)))
    if n_rename == 0:
        return out
    chosen_idx = rng.choice(len(feats), size=n_rename, replace=False)
    rename_map = {}
    for i, ci in enumerate(chosen_idx):
        old = feats[ci]
        rename_map[old] = f"col_{seed}_{i:03d}"
    return out.rename(columns=rename_map)


# ---------------- families (new realistic ones) ----------------

def currency_unit_drift(df: pd.DataFrame, severity: float, seed: int) -> pd.DataFrame:
    """Multiply `severity` fraction of monetary values by 100.

    Real-world: cents-vs-reais migration bugs, multi-currency systems where one
    feed silently switched units. Values look superficially plausible but are
    100x too large.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    for col in MONETARY_COLS:
        if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
            continue
        mask = rng.random(len(out)) < severity
        if not mask.any():
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
        out.loc[mask, col] = out.loc[mask, col] * 100.0
    return out


def timezone_collapse(df: pd.DataFrame, severity: float, seed: int) -> pd.DataFrame:
    """Shift time-derived numeric features by random hours for `severity` fraction
    of rows. Models the UTC vs BRT (-3h) confusion.

    purchase_hour shifted by an integer hour offset in {-3, +3} mod 24.
    Day-difference fields shifted by ±0.125 (3 hours expressed as days).

    Real-world: mixing UTC and local timestamps, daylight-saving boundary bugs.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    n = len(out)
    if "purchase_hour" in out.columns:
        col = "purchase_hour"
        mask = rng.random(n) < severity
        if mask.any():
            shifts = rng.choice([-3, 3], size=int(mask.sum()))
            new = pd.to_numeric(out[col], errors="coerce").astype("Int64").to_numpy(
                dtype=object
            )
            for j, sh in zip(np.where(mask)[0], shifts):
                if pd.notna(new[j]):
                    new[j] = (int(new[j]) + int(sh)) % 24
            out[col] = pd.array(new, dtype="Int64")
    for col in TIMEZONE_NUMERIC_COLS:
        if col == "purchase_hour" or col not in out.columns:
            continue
        if not pd.api.types.is_numeric_dtype(out[col]):
            continue
        mask = rng.random(n) < severity
        if not mask.any():
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
        shifts = rng.choice([-0.125, 0.125], size=int(mask.sum()))
        idxs = np.where(mask)[0]
        out.iloc[idxs, out.columns.get_loc(col)] = (
            out.iloc[idxs, out.columns.get_loc(col)].to_numpy() + shifts
        )
    return out


# Portuguese taxonomy mutations: official label -> plausibly-wrong variants
_PT_TAXONOMY_MUTATIONS: dict[str, list[str]] = {
    "bebes": ["bebê", "bebes_infantil", "bebês"],
    "moveis_decoracao": ["móveis_decoracao", "moveis_decoração", "moveis_e_decor"],
    "esporte_lazer": ["esporte_e_lazer", "esportes_lazer", "esportelazer"],
    "informatica_acessorios": ["informática_acessorios", "informatica_acessórios", "informatica_acc"],
    "telefonia": ["telefonía", "telefonia_cel", "tel"],
    "utilidades_domesticas": ["utilidades_domésticas", "util_domesticas", "utilidade_domestica"],
    "papelaria": ["papelaria_escritorio", "papelaria_e_escritorio", "papel"],
    "perfumaria": ["perfumes", "perfumaria_cosmeticos", "perfum"],
    "beleza_saude": ["beleza_e_saude", "beleza_saúde", "beleza"],
    "automotivo": ["automotivos", "auto", "automotive"],
    "cama_mesa_banho": ["cama_e_mesa_banho", "cama-mesa-banho", "cmb"],
    "brinquedos": ["brinquedo", "brinquedos_jogos", "toys"],
    "ferramentas_jardim": ["ferramentas_e_jardim", "ferramentas/jardim", "jardim_ferramentas"],
    "fashion_bolsas_e_acessorios": ["moda_bolsas_acessorios", "fashion_bolsas", "bolsas_acessorios"],
    "eletronicos": ["eletrônicos", "electronics", "eletronic"],
    "casa_construcao": ["casa_construção", "casa_e_construcao", "construcao_casa"],
    "audio": ["áudio", "audio_video", "som"],
}


def category_taxonomy_drift(df: pd.DataFrame, severity: float, seed: int) -> pd.DataFrame:
    """Mutate Portuguese category labels into plausible variants.

    e.g. 'bebes' -> 'bebê', 'eletronicos' -> 'eletrônicos'.

    Real-world: vocabulary drift over time, multilingual normalization bugs,
    inconsistent diacritics handling between source systems.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    for col in PORTUGUESE_CAT_COLS:
        if col not in out.columns:
            continue
        s = out[col].astype("object")
        mask = rng.random(len(out)) < severity
        if not mask.any():
            continue
        idxs = np.where(mask)[0]
        for j in idxs:
            v = s.iloc[j]
            if pd.isna(v):
                continue
            key = str(v)
            variants = _PT_TAXONOMY_MUTATIONS.get(key)
            if variants:
                pick = int(rng.integers(0, len(variants)))
                s.iloc[j] = variants[pick]
            else:
                s.iloc[j] = (key.replace("a", "á", 1)
                             if "a" in key else key + "_drift")
        out[col] = s
    return out


def _mojibake_one(text: str) -> str:
    """UTF-8 → Latin-1 → UTF-8 mangling. `não` → `nÃ£o`."""
    try:
        # Encode as utf-8, decode AS latin-1 (the bug), then re-encode utf-8.
        bs = text.encode("utf-8")
        misread = bs.decode("latin-1")
        return misread
    except Exception:
        return text


def mojibake_roundtrip(df: pd.DataFrame, severity: float, seed: int) -> pd.DataFrame:
    """Apply UTF-8 → Latin-1 mis-decoding to text/categorical string fields for
    `severity` fraction of rows.

    Real-world: encoding misconfiguration between source DB, ETL, and warehouse;
    famously produces strings like 'nÃ£o' instead of 'não'.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    target_cols = list(TEXT_COLS | PORTUGUESE_CAT_COLS)
    for col in target_cols:
        if col not in out.columns:
            continue
        s = out[col].astype("object")
        mask = rng.random(len(out)) < severity
        if not mask.any():
            continue
        for j in np.where(mask)[0]:
            v = s.iloc[j]
            if pd.isna(v):
                continue
            s.iloc[j] = _mojibake_one(str(v))
        out[col] = s
    return out


# ---------------- registry ----------------

REGISTRY: dict[str, Callable[[pd.DataFrame, float, int], pd.DataFrame]] = {
    # kept (5)
    "missing_injection": missing_injection,
    "type_flip": type_flip,
    "label_noise_categorical": label_noise_categorical,
    "duplicate_rows": duplicate_rows,
    "schema_drift": schema_drift,
    # new realistic (4)
    "currency_unit_drift": currency_unit_drift,
    "timezone_collapse": timezone_collapse,
    "category_taxonomy_drift": category_taxonomy_drift,
    "mojibake_roundtrip": mojibake_roundtrip,
}

FAMILIES: list[str] = list(REGISTRY.keys())


def list_families() -> list[str]:
    return list(FAMILIES)


def apply(df: pd.DataFrame, family: str, severity: float, seed: int) -> pd.DataFrame:
    if family == "clean":
        return df.copy()
    if family not in REGISTRY:
        raise KeyError(f"unknown corruption family: {family!r}; "
                       f"known: {FAMILIES}")
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity out of [0,1]: {severity}")
    return REGISTRY[family](df, severity, seed)


def apply_clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()
