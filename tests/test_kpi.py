"""KPI formula verification — hand-calculated test cases."""

import pytest


class TestTrueEfficiency:
    """TE = (P_asic + P_cooling + P_aux) / (H * eta_env * eta_mode) [J/TH]"""

    def test_te_s21_pro_normal_28c(self):
        """S21 Pro at 28C ambient, normal mode."""
        P_total = 3510 + 3510 * 0.07 + 3510 * 0.02  # 3825.9
        H = 234.0
        eta_env = max(0.70, 1.0 - 0.008 * (28 - 25))  # 0.976
        eta_mode = 1.00
        TE = P_total / (H * eta_env * eta_mode)
        assert 16.0 < TE < 18.0, f"Expected ~16.75, got {TE:.2f}"

    def test_te_s21_base_normal_25c(self):
        """S21 base at 25C (reference temp) — TE should equal raw efficiency + overhead."""
        P_total = 3500 + 3500 * 0.07 + 3500 * 0.02  # 3815
        H = 200.0
        eta_env = 1.0  # at reference temp
        eta_mode = 1.0
        TE = P_total / (H * eta_env * eta_mode)  # 19.075
        assert 18.0 < TE < 20.0, f"Expected ~19.08, got {TE:.2f}"

    def test_overclock_worse_than_normal(self):
        """Overclock eta_mode=0.85 -> TE larger (worse)."""
        P = 3825.9
        H = 234.0
        eta_env = 0.976
        te_normal = P / (H * eta_env * 1.00)
        te_oc = P / (H * eta_env * 0.85)
        assert te_oc > te_normal, f"OC {te_oc:.2f} should be > normal {te_normal:.2f}"

    def test_low_power_better_than_normal(self):
        """Low-power eta_mode=1.10 -> TE smaller (better)."""
        P = 3825.9
        H = 234.0
        eta_env = 0.976
        te_normal = P / (H * eta_env * 1.00)
        te_lp = P / (H * eta_env * 1.10)
        assert te_lp < te_normal, f"LP {te_lp:.2f} should be < normal {te_normal:.2f}"

    def test_hot_ambient_penalizes(self):
        """Higher ambient -> lower eta_env -> higher (worse) TE."""
        P = 3825.9
        H = 234.0
        eta_mode = 1.0
        eta_25 = max(0.70, 1.0 - 0.008 * (25 - 25))  # 1.0
        eta_40 = max(0.70, 1.0 - 0.008 * (40 - 25))  # 0.88
        te_25 = P / (H * eta_25 * eta_mode)
        te_40 = P / (H * eta_40 * eta_mode)
        assert te_40 > te_25, f"Hot {te_40:.2f} should be > cool {te_25:.2f}"


class TestEconomicTE:
    """ETE = (0.024 * TE * energy_price) / (hashprice / 1000) [dimensionless]"""

    def test_ete_profitable_s21_pro(self):
        """S21 Pro at $0.04/kWh, hashprice=$50 -> profitable."""
        TE = 16.75
        cost = 0.024 * TE * 0.04
        rev = 50.0 / 1000.0
        ETE = cost / rev
        assert ETE < 1.0, f"Should be profitable, ETE={ETE:.4f}"
        assert 0.2 < ETE < 0.5, f"Expected ~0.32, got {ETE:.4f}"

    def test_ete_unprofitable_expensive_power(self):
        """M60S-class at $0.15/kWh, low hashprice -> unprofitable."""
        TE = 20.0
        cost = 0.024 * TE * 0.15
        rev = 35.0 / 1000.0
        ETE = cost / rev
        assert ETE > 1.0, f"Should be unprofitable, ETE={ETE:.4f}"

    def test_ete_breakeven(self):
        """ETE=1.0 is exact breakeven. Verify math."""
        TE = 16.75
        energy_price = 0.04
        breakeven_hp = 0.024 * TE * energy_price * 1000
        ETE = (0.024 * TE * energy_price) / (breakeven_hp / 1000)
        assert abs(ETE - 1.0) < 0.001, f"Breakeven ETE should be 1.0, got {ETE:.6f}"


class TestProfitDensity:
    """PD = (daily_revenue - daily_cost) / P_total [$/W/day]"""

    def test_profit_density_positive_when_profitable(self):
        P_total = 3825.9
        H = 234.0
        eta_env = 0.976
        eta_mode = 1.0
        energy_price = 0.04
        hashprice = 50.0

        daily_rev = hashprice / 1000.0 * H
        daily_cost = 0.024 * P_total / (eta_env * eta_mode) * energy_price
        PD = (daily_rev - daily_cost) / P_total
        assert PD > 0, f"Should be positive, PD={PD:.6f}"

    def test_profit_density_negative_when_unprofitable(self):
        P_total = 3825.9
        H = 186.0  # M60S
        eta_env = 0.88
        eta_mode = 1.0
        energy_price = 0.15
        hashprice = 30.0

        daily_rev = hashprice / 1000.0 * H
        daily_cost = 0.024 * P_total / (eta_env * eta_mode) * energy_price
        PD = (daily_rev - daily_cost) / P_total
        assert PD < 0, f"Should be negative, PD={PD:.6f}"
