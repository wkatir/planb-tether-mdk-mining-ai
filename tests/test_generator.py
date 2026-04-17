"""Tests for synthetic data generator."""

import numpy as np
import pytest


class TestSyntheticDataGenerator:
    def test_generator_creates(self):
        from app.data.generator import SyntheticDataGenerator

        gen = SyntheticDataGenerator(fleet_size=5, days=1, seed=42)
        assert gen is not None
        assert gen.fleet_size == 5
        assert gen.total_steps > 0

    def test_ambient_range(self):
        from app.data.generator import SyntheticDataGenerator

        gen = SyntheticDataGenerator(fleet_size=2, days=1, seed=42)
        temps = gen._generate_ambient()
        assert np.all(temps >= 5.0), f"Min too low: {temps.min()}"
        assert np.all(temps <= 50.0), f"Max too high: {temps.max()}"

    def test_energy_price_range(self):
        from app.data.generator import SyntheticDataGenerator

        gen = SyntheticDataGenerator(fleet_size=2, days=1, seed=42)
        prices = gen._generate_energy_price()
        assert np.all(prices >= 0.02), f"Min too low: {prices.min()}"
        assert np.all(prices <= 0.50), f"Max too high: {prices.max()}"

    def test_hashprice_range(self):
        from app.data.generator import SyntheticDataGenerator

        gen = SyntheticDataGenerator(fleet_size=2, days=1, seed=42)
        hp = gen._generate_hashprice()
        assert np.all(hp >= 20.0), f"Min too low: {hp.min()}"
        assert np.all(hp <= 120.0), f"Max too high: {hp.max()}"

    def test_fleet_includes_s21_pro(self):
        from app.data.generator import SyntheticDataGenerator

        gen = SyntheticDataGenerator(fleet_size=50, days=1, seed=42)
        models = [m.spec.model_name for m in gen.miners]
        assert "Antminer S21 Pro" in models, f"S21 Pro missing. Models: {set(models)}"

    def test_failure_injection_present(self):
        from app.data.generator import SyntheticDataGenerator

        gen = SyntheticDataGenerator(fleet_size=50, days=1, seed=42)
        unhealthy = [m for m in gen.miners if not m.is_healthy]
        assert len(unhealthy) >= 1, "Expected at least 1 unhealthy miner"
        types = {m.failure_type for m in unhealthy}
        assert types.issubset({"thermal", "hashboard", "psu"}), f"Bad types: {types}"
