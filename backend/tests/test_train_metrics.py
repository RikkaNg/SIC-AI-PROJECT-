"""
test_train_metrics.py - Unit test bộ metrics + contract feature preprocessor
=============================================================================
1. train.evaluate(): đủ 6 metrics, khớp công thức evaluate_metrics.py.
2. engineer_features(): tạo đủ cột theo COLS_FILL_ZERO/COLS_CATEGORICAL/
   COLS_PASSTHROUGH; công thức rolling std + payday đúng trên dữ liệu giả.

Chạy: pytest backend/tests/test_train_metrics.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml_training" / "src"))

from train import evaluate
from preprocessor import engineer_features, COLS_FILL_ZERO, COLS_CATEGORICAL, COLS_PASSTHROUGH
from cluster_features import ClusterFeatureEngineer


class TestEvaluateMetrics:
    """train.evaluate() trả đủ bộ 6 metrics và khớp công thức chuẩn."""

    Y = np.array([2.0, 0.0, 5.0, 10.0, 3.0, 7.0])
    PRED = np.array([2.5, 1.0, 4.0, 8.0, 3.0, 9.0])
    PERI = np.array([1, 0, 1, 0, 1, 0])

    def test_has_all_six_metrics(self):
        m = evaluate(self.Y, self.PRED, self.PERI)
        for key in ("rmsle", "mae", "rmse", "wape", "wmape", "r2"):
            assert key in m, f"thiếu metric {key}"
            assert m[key] is not None

    def test_mae_rmse_manual(self):
        m = evaluate(self.Y, self.PRED)
        err = self.PRED - self.Y
        assert m["mae"] == pytest.approx(np.abs(err).mean())
        assert m["rmse"] == pytest.approx(np.sqrt((err ** 2).mean()))

    def test_wmape_equals_wape_without_weights(self):
        """Không có perishable → w=1 → WMAPE ≡ WAPE."""
        m = evaluate(self.Y, self.PRED)
        assert m["wmape"] == pytest.approx(m["wape"])

    def test_wmape_weighted_manual(self):
        m = evaluate(self.Y, self.PRED, self.PERI)
        err = np.abs(self.PRED - self.Y)
        w = np.where(self.PERI > 0, 1.5, 1.0)
        expected = (w * err).sum() / (w * self.Y).sum() * 100
        assert m["wmape"] == pytest.approx(expected)

    def test_rmsle_manual(self):
        m = evaluate(self.Y, self.PRED)
        expected = np.sqrt(np.mean((np.log1p(self.PRED) - np.log1p(self.Y)) ** 2))
        assert m["rmsle"] == pytest.approx(expected)

    def test_perfect_prediction(self):
        m = evaluate(self.Y, self.Y.copy(), self.PERI)
        assert m["rmsle"] == pytest.approx(0.0)
        assert m["mae"] == pytest.approx(0.0)
        assert m["wmape"] == pytest.approx(0.0)
        assert m["r2"] == pytest.approx(1.0)

    def test_negative_preds_clipped(self):
        """Dự báo âm bị clip về 0 — không sinh NaN/Inf."""
        m = evaluate(self.Y, np.array([-5.0, -1.0, 4.0, 8.0, 3.0, 9.0]), self.PERI)
        for key in ("rmsle", "mae", "rmse", "wape", "wmape"):
            assert np.isfinite(m[key])


def _make_synthetic_df(n_days: int = 45) -> pd.DataFrame:
    """2 cửa hàng × 2 ngành × n_days ngày, đủ cột đầu vào cho engineer_features."""
    rng = np.random.default_rng(42)
    rows = []
    for store in (1, 2):
        for family in ("GROCERY I", "BEVERAGES"):
            base = 10.0 * store + (5.0 if family == "BEVERAGES" else 0.0)
            for i, d in enumerate(pd.date_range("2017-06-01", periods=n_days)):
                rows.append({
                    "date": d,
                    "store_nbr": store,
                    "family": family,
                    "target": max(0.0, base + rng.normal(0, 2.0)),
                    "onpromotion": int(i % 7 == 0),
                    "is_earthquake_period": 0,
                    "holiday_type": "Normal Day",
                    "oil_price": 50.0,
                    "perishable": 1 if family == "BEVERAGES" else 0,
                    "transactions": 800.0 + 10.0 * store + rng.normal(0, 20.0),
                    "city": "Quito" if store == 1 else "Guayaquil",
                    "state": "Pichincha" if store == 1 else "Guayas",
                    "type": "D",
                    "cluster": 5 if store == 1 else 11,
                })
    return pd.DataFrame(rows)


class TestEngineerFeaturesContract:
    """engineer_features tạo đủ cột theo 3 danh sách COLS_* và công thức đúng."""

    @pytest.fixture(scope="class")
    def featured(self):
        df = _make_synthetic_df()
        ce = ClusterFeatureEngineer(smoothing=10.0)
        ce.fit(df, target_col="target")
        return engineer_features(df, cluster_engineer=ce)

    def test_all_schema_columns_present(self, featured):
        expected = COLS_FILL_ZERO + COLS_CATEGORICAL + COLS_PASSTHROUGH
        missing = [c for c in expected if c not in featured.columns]
        assert not missing, f"Thiếu cột sau engineering: {missing}"

    def test_rolling_std_formula(self, featured):
        """sales_std7 = std mẫu (ddof=1) của 7 ngày liền trước: shift(1).rolling(7)."""
        for key, grp in featured.groupby(["store_nbr", "family"]):
            grp = grp.sort_values("date").reset_index(drop=True)
            targets = grp["target"].to_numpy()
            i = 12  # dòng đủ history
            expected = np.std(targets[i - 7:i], ddof=1)  # 7 ngày trước, không gồm ngày i
            assert grp["sales_std7"].iloc[i] == pytest.approx(expected, rel=1e-6)

    def test_rolling_mean30_exists_and_shifted(self, featured):
        """sales_rolling_mean30 tại dòng cuối phải KHÔNG chứa target của chính ngày đó."""
        grp = (featured.groupby(["store_nbr", "family"])
               .get_group((1, "GROCERY I")).sort_values("date").reset_index(drop=True))
        i = len(grp) - 1
        window = grp["target"].iloc[i - 30:i]  # 30 ngày trước, không gồm ngày i
        assert grp["sales_rolling_mean30"].iloc[i] == pytest.approx(window.mean(), rel=1e-6)

    def test_payday_flags(self, featured):
        """is_payday đúng ngày 15 + ngày cuối tháng; days_to/from_payday đúng mốc."""
        df = featured
        d = df["date"].dt
        assert (df["is_payday"] == ((d.day == 15) | (d.day == d.days_in_month)).astype(int)).all()
        # Ngày 10 → còn 5 ngày tới kỳ lương 15; ngày 20 → đã qua kỳ lương 15 ước 5 ngày
        row10 = df[d.day == 10].iloc[0]
        row20 = df[d.day == 20].iloc[0]
        assert row10["days_to_payday"] == 5
        assert row20["days_from_payday"] == 5

    def test_no_future_leak_in_lag_features(self, featured):
        """Các cột phụ thuộc target tại dòng i không đổi khi đổi target của ngày i."""
        df = _make_synthetic_df().copy()
        f1 = engineer_features(df)
        df_mod = df.copy()
        df_mod.loc[df_mod.index[-1], "target"] = 99999.0
        f2 = engineer_features(df_mod)
        last1 = f1.iloc[-1]
        last2 = f2.iloc[-1]
        for col in ("sales_lag7", "sales_lag14", "sales_lag28", "sales_rolling_mean7",
                    "sales_rolling_mean14", "sales_rolling_mean30", "sales_std7", "sales_std28"):
            v1, v2 = last1[col], last2[col]
            if pd.isna(v1) and pd.isna(v2):
                continue
            assert v1 == pytest.approx(v2), f"{col} bị ảnh hưởng bởi target cùng ngày (leak)"
