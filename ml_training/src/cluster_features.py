"""
src/cluster_features.py
Khai thác biến cluster: Target Encoding + Interaction Features.
Tất cả target encoding được tính CHỈ trên train để tránh data leakage.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class ClusterFeatureEngineer:
    """
    Tính toán các đặc trưng dựa trên cluster.
    Fit trên train → transform trên train/val/test.
    """

    def __init__(self, smoothing: float = 10.0):
        self.smoothing = smoothing
        self.global_mean = None
        self.cluster_stats = None
        self.cluster_family_stats = None
        self.cluster_promo_stats = None
        self.tier1_clusters = {5, 11}

    def fit(self, df: pd.DataFrame, target_col: str = "target") -> "ClusterFeatureEngineer":
        """Tính target encoding CHỈ trên train."""
        self.global_mean = df[target_col].mean()

        # 1. Cluster-level target encoding (smoothed)
        cluster_agg = df.groupby("cluster")[target_col].agg(["mean", "median", "std", "count"])
        cluster_agg["mean_smooth"] = (
            cluster_agg["mean"] * cluster_agg["count"] + self.global_mean * self.smoothing
        ) / (cluster_agg["count"] + self.smoothing)
        cluster_agg["median_smooth"] = (
            cluster_agg["median"] * cluster_agg["count"] + self.global_mean * self.smoothing
        ) / (cluster_agg["count"] + self.smoothing)
        self.cluster_stats = cluster_agg[["mean_smooth", "median_smooth", "std"]].rename(
            columns={
                "mean_smooth": "cluster_mean_sales",
                "median_smooth": "cluster_median_sales",
                "std": "cluster_std_sales",
            }
        )

        # 2. Cluster × Family interaction
        cf_agg = df.groupby(["cluster", "family"])[target_col].agg(["mean", "count"])
        cf_agg["mean_smooth"] = (
            cf_agg["mean"] * cf_agg["count"] + self.global_mean * self.smoothing
        ) / (cf_agg["count"] + self.smoothing)
        self.cluster_family_stats = cf_agg[["mean_smooth"]].rename(
            columns={"mean_smooth": "cluster_family_mean_sales"}
        )

        # 3. Cluster × onpromotion interaction
        cp_agg = df.groupby(["cluster", "onpromotion"])[target_col].agg(["mean", "count"])
        cp_agg["mean_smooth"] = (
            cp_agg["mean"] * cp_agg["count"] + self.global_mean * self.smoothing
        ) / (cp_agg["count"] + self.smoothing)
        self.cluster_promo_stats = cp_agg[["mean_smooth"]].rename(
            columns={"mean_smooth": "cluster_promo_mean_sales"}
        )

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gán các đặc trưng đã tính vào dataframe + tính interaction features."""
        df = df.copy()

        # 1. Cluster-level stats
        df = df.merge(self.cluster_stats, left_on="cluster", right_index=True, how="left")

        # 2. Cluster × Family
        df = df.merge(
            self.cluster_family_stats,
            left_on=["cluster", "family"],
            right_index=True,
            how="left",
        )

        # 3. Cluster × onpromotion
        df = df.merge(
            self.cluster_promo_stats,
            left_on=["cluster", "onpromotion"],
            right_index=True,
            how="left",
        )

        # 4. Tier flag
        df["is_tier1_cluster"] = df["cluster"].isin(self.tier1_clusters).astype(int)

        # 5. Interaction features (tính từ các stats vừa merge)
        # Promo lift: tỷ lệ doanh số khi có promotion vs không có promotion
        if "cluster_promo_mean_sales" in df.columns and "cluster_mean_sales" in df.columns:
            df["cluster_promo_lift"] = (
                df["cluster_promo_mean_sales"] / (df["cluster_mean_sales"] + 1e-8)
            )

        # Fillna cho các cluster/family/promo combo chưa thấy trong train
        fill_cols = [
            "cluster_mean_sales",
            "cluster_median_sales",
            "cluster_std_sales",
            "cluster_family_mean_sales",
            "cluster_promo_mean_sales",
            "cluster_promo_lift",
        ]
        for col in fill_cols:
            if col in df.columns:
                df[col] = df[col].fillna(self.global_mean)

        return df

    def fit_transform(self, df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
        return self.fit(df, target_col).transform(df)


def split_by_tier(df: pd.DataFrame, tier1_clusters: set = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chia dataframe thành Tier 1 và Tier 2+ theo cluster."""
    if tier1_clusters is None:
        tier1_clusters = {5, 11}
    mask_tier1 = df["cluster"].isin(tier1_clusters)
    return df[mask_tier1].copy(), df[~mask_tier1].copy()