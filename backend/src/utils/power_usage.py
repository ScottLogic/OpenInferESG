import json
import logging
import os

logger = logging.getLogger(__name__)

JSON_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "openai_model_specs.json"))
openai_power_stats = None
model_power_dict = None

def calculate_model_power(model: str, openai_power_stats: dict[str, dict]) -> float | None:
    model_stats = openai_power_stats.get(model)
    if model_stats is None:
        return None

    gpu_power = model_stats["gpu_power"]
    non_gpu_power = model_stats["non_gpu_power"]
    pue = model_stats["pue"]
    assigned_gpus = model_stats["assigned_gpus"]
    gpus_per_node = model_stats["gpus_per_node"]
    batch_size = model_stats["batch_size"]
    non_gpu_draw = model_stats["non_gpu_draw"]
    gpu_draw = model_stats["gpu_draw"]

    total_gpu_utilisation = (assigned_gpus * gpu_draw) / (gpus_per_node * batch_size)
    total_non_gpu_utilisation = (assigned_gpus * non_gpu_draw) / (gpus_per_node * batch_size)
    effective_power = ((gpu_power * total_gpu_utilisation) + (non_gpu_power * total_non_gpu_utilisation)) * pue

    return effective_power

def calculate_power_usage(duration_seconds: float, model: str) -> float | None:
    global model_power_dict
    if model_power_dict is None:
        openai_power_stats = json.load(open(JSON_FILE_PATH))
        model_power_dict = dict(map(lambda key: (key, calculate_model_power(key, openai_power_stats)), openai_power_stats.keys()))

    model_power = model_power_dict.get(model)
    if model_power is None:
        logger.warning(f"Power details not found for model: {model}")
        return None

    duration_hours = duration_seconds / 3600
    return duration_hours * model_power * 1000  # converting from kWh to Wh
