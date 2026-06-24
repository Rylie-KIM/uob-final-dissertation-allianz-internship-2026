import pandas as pd

from generate.enrichment import BASE_REPAIR_COST


class EnrichmentImputer:
    """
    Fit on training data; apply identical transform at inference.

    Prevents training-serving skew in two places:
      1. Enrichment join misses (old vehicles not in YEAR_BANDS) — filled with
         group medians keyed by (vehicle_make, vehicle_model).
      2. is_high_value_vehicle threshold — stored from training data so inference
         always uses the same cut-off, not a recomputed one.
    """

    def __init__(self) -> None:
        self._group_medians: pd.DataFrame | None = None
        self._global_market_value: float | None = None
        self._global_part_index: float | None = None
        self._high_value_threshold: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, enrichment_table: pd.DataFrame, train_df: pd.DataFrame) -> "EnrichmentImputer":
        """
        Compute and store imputation statistics from the enrichment table and
        training-window rows.

        Args:
            enrichment_table: output of build_enrichment_table() — used for
                              group medians; static reference, no leakage.
            train_df:         rows belonging to the v1 training window — used
                              for the high-value threshold to avoid future leakage.
        """
        self._group_medians = (
            enrichment_table
            .groupby(["vehicle_make", "vehicle_model"])[
                ["typical_market_value_gbp", "part_cost_index"]
            ]
            .median()
        )
        self._global_market_value = float(enrichment_table["typical_market_value_gbp"].median())
        self._global_part_index   = float(enrichment_table["part_cost_index"].median())

        # Compute threshold on training data *after* filling nulls so the
        # distribution reflects what the model actually sees.
        df_temp = self._fill_nulls(train_df.copy())
        self._high_value_threshold = float(df_temp["vehicle_value"].quantile(0.75))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill enrichment nulls and rewrite is_high_value_vehicle using the
        stored training threshold."""
        df = df.copy()
        df = self._fill_nulls(df)
        df["is_high_value_vehicle"] = df["vehicle_value"] > self._high_value_threshold
        return df

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fill_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        null_mask = df["typical_market_value_gbp"].isna()
        if not null_mask.any():
            return df

        # Vectorised: merge group medians onto null rows
        null_rows = df.loc[null_mask, ["vehicle_make", "vehicle_model"]].copy()
        filled = null_rows.merge(
            self._group_medians.reset_index(),
            on=["vehicle_make", "vehicle_model"],
            how="left",
        )
        filled.index = null_rows.index

        for col, fallback in [
            ("typical_market_value_gbp", self._global_market_value),
            ("part_cost_index",          self._global_part_index),
        ]:
            df.loc[null_mask, col] = filled[col].fillna(fallback).values

        # Recompute downstream derived columns only for the imputed rows
        age = df.loc[null_mask, "vehicle_age_years"]
        depreciation = (1.0 - 0.08 * age).clip(lower=0.10)
        df.loc[null_mask, "vehicle_value"] = (
            df.loc[null_mask, "typical_market_value_gbp"] * depreciation
        ).round(2)

        base_cost = df.loc[null_mask].apply(
            lambda r: BASE_REPAIR_COST.get((r["damage_severity"], r["damage_location"]), 1_000),
            axis=1,
        )
        df.loc[null_mask, "repair_estimate_gbp"] = (
            df.loc[null_mask, "part_cost_index"] * base_cost
        ).round(2)

        df.loc[null_mask, "repair_to_value_ratio"] = (
            df.loc[null_mask, "repair_estimate_gbp"] / df.loc[null_mask, "vehicle_value"]
        ).clip(upper=2.0).round(4)

        return df
