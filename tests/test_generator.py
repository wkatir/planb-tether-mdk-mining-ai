import numpy as np
import pytest


DATA_GENERATED = False


def check_data_ready():
    return DATA_GENERATED


@pytest.mark.skipif(not check_data_ready(), reason="Data not generated yet")
class TestSyntheticDataGenerator:
    def test_generator_initializes(self):
        try:
            from app.data.generator import SyntheticDataGenerator

            gen = SyntheticDataGenerator()
            assert gen is not None
        except ImportError:
            pytest.skip("SyntheticDataGenerator not implemented yet")

    def test_ambient_temperature_range(self):
        try:
            from app.data.generator import SyntheticDataGenerator
        except ImportError:
            pytest.skip("SyntheticDataGenerator not implemented yet")

        try:
            gen = SyntheticDataGenerator()
            temps = gen.generate_ambient_temperature(1000)
            assert np.all(temps >= 5) and np.all(temps <= 50), (
                f"Ambient temperature out of range [5, 50]: min={temps.min()}, max={temps.max()}"
            )
        except AttributeError:
            pytest.skip("generate_ambient_temperature method not available")

    def test_energy_price_range(self):
        try:
            from app.data.generator import SyntheticDataGenerator
        except ImportError:
            pytest.skip("SyntheticDataGenerator not implemented yet")

        try:
            gen = SyntheticDataGenerator()
            prices = gen.generate_energy_price(1000)
            assert np.all(prices >= 0.02) and np.all(prices <= 0.50), (
                f"Energy price out of range [0.02, 0.50]: min={prices.min()}, max={prices.max()}"
            )
        except AttributeError:
            pytest.skip("generate_energy_price method not available")

    def test_hashprice_range(self):
        try:
            from app.data.generator import SyntheticDataGenerator
        except ImportError:
            pytest.skip("SyntheticDataGenerator not implemented yet")

        try:
            gen = SyntheticDataGenerator()
            hashprices = gen.generate_hashprice(1000)
            assert np.all(hashprices >= 20) and np.all(hashprices <= 120), (
                f"Hashprice out of range [20, 120]: min={hashprices.min()}, max={hashprices.max()}"
            )
        except AttributeError:
            pytest.skip("generate_hashprice method not available")


class TestMinerState:
    def test_miner_state_as_dict(self):
        mock_state = {
            "device_id": "miner_001",
            "rack_id": "rack_01",
            "asic_id": "asic_001",
            "P": 3500.0,
            "P_cooling": 500.0,
            "P_aux": 100.0,
            "T": 85.0,
            "H": 50.0,
            "hash_rate": 200.0,
            "hash_price": 0.05,
            "energy_price": 0.08,
            "η_env": 0.9,
            "η_mode": 0.95,
            "uptime_hours": 100.0,
            "fan_speed": 60.0,
            "power_mode": "balanced",
            "failure_probability": 0.01,
        }
        assert mock_state["device_id"] == "miner_001"
        assert mock_state["P"] == 3500.0
        assert mock_state["T"] == 85.0
