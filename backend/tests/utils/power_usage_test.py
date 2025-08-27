import pytest

# Patch os.path.abspath and os.path.dirname to avoid filesystem dependency
import src.utils.power_usage as power_usage

@pytest.fixture
def mock_openai_power_stats():
    return {
        "test-model": {
            "gpu_power": 0.5,
            "non_gpu_power": 0.2,
            "pue": 1.2,
            "assigned_gpus": 2,
            "gpus_per_node": 4,
            "batch_size": 8,
            "non_gpu_draw": 0.1,
            "gpu_draw": 0.3
        }
    }

def test_calculate_model_power_returns_expected_value(mock_openai_power_stats):
    result = power_usage.calculate_model_power("test-model", mock_openai_power_stats)
    # Manual calculation:
    # total_gpu_utilisation = (2*0.3)/(4*8) = 0.01875 = 0.01875
    # total_non_gpu_utilisation = (2*0.1)/(4*8) = 0.00625
    # effective_power = ((0.5*0.01875)+(0.2*0.00625))*1.2 = (0.009375+0.00125)*1.2 = 0.01275
    assert pytest.approx(result, 0.0001) == 0.01275

def test_calculate_model_power_returns_none_if_stats_none():
    stats = {"test-model-2": {}}
    assert power_usage.calculate_model_power("test-model", stats) is None

def test_calculate_power_usage_returns_expected(monkeypatch, mock_openai_power_stats):
    # Reset global cache
    power_usage.model_power_dict = None
    monkeypatch.setattr(power_usage.json, "load", lambda f: mock_openai_power_stats)
    duration_seconds = 3600  # 1 hour
    result = power_usage.calculate_power_usage(duration_seconds, "test-model")
    # model_power = 0.01275, duration_hours = 1, result = 1*0.01275*1000 = 12.75
    assert pytest.approx(result, 0.01) == 12.75

def test_calculate_power_usage_returns_none_for_unknown_model(monkeypatch, mock_openai_power_stats, caplog):
    monkeypatch.setattr(power_usage.json, "load", lambda f: mock_openai_power_stats)
    power_usage.model_power_dict = None
    with caplog.at_level("WARNING"):
        result = power_usage.calculate_power_usage(100, "unknown-model")
        assert result is None
        assert "Power details not found for model: unknown-model" in caplog.text

def test_calculate_power_usage_uses_cached_model_power_dict(monkeypatch, mock_openai_power_stats):
    # Set up the cache
    power_usage.model_power_dict = {"test-model": 0.0153}
    result = power_usage.calculate_power_usage(3600, "test-model")
    assert pytest.approx(result, 0.01) == 15.3