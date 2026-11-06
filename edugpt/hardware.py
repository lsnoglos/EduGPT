"""Detección de hardware y perfiles locales; no se consultan servicios externos."""
from __future__ import annotations

import json
import os
from pathlib import Path

import psutil
import torch

PROFILES = ("Ahorrador", "Equilibrado", "Alto rendimiento")
DEVICE_AUTO = "auto"


def detect_hardware() -> dict:
    """Consulta los recursos del EQUIPO donde se ejecuta EduGPT."""
    ram = psutil.virtual_memory()
    gpu_devices = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            try:
                free_bytes, _ = torch.cuda.mem_get_info(idx)
            except RuntimeError:
                free_bytes = None
            gpu_devices.append({
                "id": f"cuda:{idx}",
                "name": props.name,
                "total_gb": round(props.total_memory / (1024 ** 3), 2),
                "free_gb": round(free_bytes / (1024 ** 3), 2) if free_bytes is not None else None,
            })
    return {
        "cpu_logical_cores": max(1, os.cpu_count() or 1),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "ram_total_gb": round(ram.total / (1024 ** 3), 2),
        "ram_available_gb": round(ram.available / (1024 ** 3), 2),
        "gpu_devices": gpu_devices,
    }


def device_options(hardware: dict) -> list[str]:
    return [DEVICE_AUTO, "cpu"] + [gpu["id"] for gpu in hardware["gpu_devices"]]


def resolve_device(requested: str = DEVICE_AUTO) -> torch.device:
    """La GPU solicitada nunca se sustituye silenciosamente por CPU."""
    if requested == DEVICE_AUTO:
        return torch.device("cuda:0" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda:"):
        try:
            idx = int(requested.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError("Dispositivo CUDA inválido.") from exc
        if not torch.cuda.is_available() or not 0 <= idx < torch.cuda.device_count():
            raise ValueError(f"El dispositivo {requested} no está disponible. Instala PyTorch con CUDA o elige CPU.")
        return torch.device(requested)
    raise ValueError(f"Dispositivo desconocido: {requested}")


def profile_defaults(hardware: dict, profile: str = "Equilibrado") -> dict:
    if profile not in PROFILES:
        raise ValueError(f"Perfil desconocido: {profile}")
    has_gpu = bool(hardware["gpu_devices"])
    cores = hardware["cpu_logical_cores"]
    base = {
        "profile": profile,
        "device": DEVICE_AUTO,
        "cpu_threads": max(1, min(cores, max(1, cores // 2))),
        "gpu_memory_fraction": 0.80,
        "dimensions": 64,
        "window": 2,
        "epochs": 10,
        "negatives": 5,
        "batch_size": 512 if has_gpu else 128,
        "max_pairs": 150000 if has_gpu else 50000,
        "learning_rate": 0.003,
        "seed": 42,
        "snapshot_every": 5,
    }
    if profile == "Ahorrador":
        base.update(cpu_threads=max(1, min(2, cores)), gpu_memory_fraction=0.60,
                    dimensions=32, epochs=5, negatives=3, batch_size=128 if has_gpu else 64,
                    max_pairs=30000, snapshot_every=5)
    elif profile == "Alto rendimiento":
        base.update(cpu_threads=max(1, min(cores, max(1, cores - 1))),
                    gpu_memory_fraction=0.90, dimensions=128, epochs=20, negatives=8,
                    batch_size=1024 if has_gpu else 256, max_pairs=300000,
                    snapshot_every=2)
    # Ajustar presets al hardware, sin imponer límites irreales al modo manual.
    if hardware["ram_total_gb"] < 8:
        base["max_pairs"] = min(base["max_pairs"], 30000)
        base["batch_size"] = min(base["batch_size"], 64)
        base["dimensions"] = min(base["dimensions"], 64)
    if has_gpu and max(g["total_gb"] for g in hardware["gpu_devices"]) < 6:
        base["batch_size"] = min(base["batch_size"], 256)
        base["dimensions"] = min(base["dimensions"], 64)
    return base


def validate_settings(settings: dict, hardware: dict) -> dict:
    """Validación estricta antes de persistir o iniciar entrenamiento."""
    cfg = dict(settings)
    if cfg.get("profile") not in PROFILES:
        raise ValueError("Perfil inválido.")
    if cfg.get("device") not in device_options(hardware):
        raise ValueError("La CPU/GPU seleccionada no está disponible en este equipo.")
    ints = {
        "cpu_threads": (1, hardware["cpu_logical_cores"]),
        "dimensions": (8, 512),
        "window": (1, 10),
        "epochs": (1, 200),
        "negatives": (1, 20),
        "batch_size": (16, 4096),
        "max_pairs": (100, 1000000),
        "seed": (0, 2147483647),
        "snapshot_every": (1, 200),
    }
    for name, (minimum, maximum) in ints.items():
        value = cfg.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f"{name} debe ser un entero entre {minimum} y {maximum}.")
    fraction = cfg.get("gpu_memory_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0.4 <= fraction <= 0.95:
        raise ValueError("gpu_memory_fraction debe estar entre 0.40 y 0.95.")
    rate = cfg.get("learning_rate")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0.00001 <= rate <= 0.1:
        raise ValueError("learning_rate debe estar entre 0.00001 y 0.1.")
    return cfg


def load_settings(path: Path, hardware: dict) -> dict:
    defaults = profile_defaults(hardware)
    if not path.exists():
        return defaults
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        # No cargamos dispositivos de GPU inexistentes al cambiar de equipo.
        candidate = {**defaults, **stored}
        if candidate["device"] not in device_options(hardware):
            candidate["device"] = DEVICE_AUTO
        candidate["cpu_threads"] = min(candidate["cpu_threads"], hardware["cpu_logical_cores"])
        return validate_settings(candidate, hardware)
    except (OSError, ValueError, TypeError, KeyError):
        return defaults


def save_settings(path: Path, settings: dict, hardware: dict) -> dict:
    clean = validate_settings(settings, hardware)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)
    return clean
