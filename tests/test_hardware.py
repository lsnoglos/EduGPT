import json

import pytest

from edugpt.hardware import device_options, load_settings, profile_defaults, resolve_device, save_settings
from edugpt.core import prepare_corpus
from edugpt.trainer import train


@pytest.fixture
def hardware():
    return {
        "cpu_logical_cores": 8, "cpu_physical_cores": 4,
        "ram_total_gb": 16.0, "ram_available_gb": 12.0,
        "gpu_devices": [{"id": "cuda:0", "name": "RTX de prueba",
                         "total_gb": 8.0, "free_gb": 7.0}],
    }


def test_profiles_scale_resource_usage(hardware):
    economy = profile_defaults(hardware, "Ahorrador")
    balanced = profile_defaults(hardware, "Equilibrado")
    performance = profile_defaults(hardware, "Alto rendimiento")
    assert economy["batch_size"] < balanced["batch_size"] < performance["batch_size"]
    assert economy["max_pairs"] < performance["max_pairs"]
    assert economy["gpu_memory_fraction"] < performance["gpu_memory_fraction"]
    assert all(1 <= cfg["cpu_threads"] <= 8 for cfg in (economy, balanced, performance))


def test_settings_roundtrip_and_cpu_override(tmp_path, hardware):
    path = tmp_path / "settings.json"
    config = profile_defaults(hardware)
    config.update(device="cpu", batch_size=64, cpu_threads=2, epochs=3)
    save_settings(path, config, hardware)
    loaded = load_settings(path, hardware)
    assert loaded == config
    assert resolve_device(loaded["device"]).type == "cpu"
    assert device_options(hardware) == ["auto", "cpu", "cuda:0"]


def test_settings_invalid_values_not_saved(tmp_path, hardware):
    path = tmp_path / "settings.json"
    invalid = profile_defaults(hardware)
    invalid["cpu_threads"] = 200
    with pytest.raises(ValueError):
        save_settings(path, invalid, hardware)
    assert not path.exists()


def test_unavailable_explicit_gpu_is_an_error():
    with pytest.raises(ValueError, match="no está disponible"):
        resolve_device("cuda:9999")


def test_training_respects_cpu_and_resource_parameters(tmp_path):
    books = tmp_path / "books"
    books.mkdir()
    (books / "base.txt").write_text(
        ("la escuela enseña el estudiante aprende el maestro explica " * 12),
        encoding="utf-8",
    )
    corpus = prepare_corpus(books, tmp_path / "corpus.json")
    run = tmp_path / "run"
    metrics = train(corpus, run, device="cpu", cpu_threads=1, epochs=2,
                    dimensions=16, batch_size=16, max_pairs=150,
                    negatives=2, snapshot_every=2)
    saved_config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert saved_config["device"] == "cpu"
    assert saved_config["cpu_threads"] == 1
    assert saved_config["batch_size"] == 16
    assert saved_config["snapshot_every"] == 2
    assert "cpu_process_ram_gb" in metrics[0]
    assert metrics[0]["gpu_peak_allocated_gb"] == 0.0
