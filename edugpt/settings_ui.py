"""Controles de hardware y entrenamiento, compartidos entre primer uso y ajustes."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from edugpt.hardware import (
    PROFILES, detect_hardware, device_options, load_settings, profile_defaults, save_settings
)


def show_hardware(hardware: dict) -> None:
    a, b, c = st.columns(3)
    a.metric("Núcleos lógicos CPU", hardware["cpu_logical_cores"])
    b.metric("RAM disponible", f'{hardware["ram_available_gb"]:.1f} GB')
    c.metric("RAM instalada", f'{hardware["ram_total_gb"]:.1f} GB')
    if hardware["gpu_devices"]:
        for gpu in hardware["gpu_devices"]:
            st.info(f'🎮 {gpu["name"]} ({gpu["id"]}) · {gpu["total_gb"]:.1f} GB VRAM'
                    + (f' · {gpu["free_gb"]:.1f} GB libres ahora' if gpu["free_gb"] is not None else ""))
    else:
        st.warning("No hay GPU CUDA disponible para PyTorch. Puedes entrenar en CPU. "
                   "Si tienes NVIDIA, revisa controladores y la instalación de PyTorch con CUDA.")


def settings_editor(settings_path: Path, hardware: dict, first_time: bool = False) -> None:
    """Form de configuración persistente; los presets se ajustan al hardware detectado."""
    if first_time:
        st.subheader("Configuración inicial de EduGPT")
        st.write("Detectamos los recursos de esta computadora. Puedes aceptar la configuración "
                 "recomendada o modificarla antes del primer entrenamiento.")
    else:
        st.subheader("Configuración de hardware y entrenamiento")
        st.write("Estos ajustes se guardan en esta computadora y se aplican al siguiente entrenamiento.")
    show_hardware(hardware)
    saved = load_settings(settings_path, hardware)
    profile = st.selectbox(
        "Perfil de consumo",
        PROFILES, index=PROFILES.index(saved["profile"]), key="edugpt_profile_selector",
        help="Ahorrador usa menos recursos; Equilibrado es el recomendado; Alto rendimiento aumenta la carga."
    )
    # El cambio de perfil carga los valores recomendados para ese perfil.
    defaults = saved if profile == saved["profile"] else profile_defaults(hardware, profile)
    devices = device_options(hardware)
    labels = {"auto": "Automático (GPU si está disponible; en otro caso CPU)", "cpu": "Solo CPU"}
    labels.update({g["id"]: f'GPU · {g["name"]} ({g["id"]})' for g in hardware["gpu_devices"]})
    with st.form("edugpt_resources_form"):
        st.markdown("#### Procesamiento")
        device = st.selectbox("Dispositivo para entrenar", devices,
                              index=devices.index(defaults["device"]), format_func=lambda val: labels[val],
                              help="Si eliges CPU, jamás se usará la GPU. Si eliges CUDA y no está disponible, el entrenamiento se detiene.")
        cpu_threads = st.slider("Hilos de CPU", min_value=1, max_value=hardware["cpu_logical_cores"],
                                value=min(defaults["cpu_threads"], hardware["cpu_logical_cores"]),
                                help="Número de hilos de cálculo CPU que puede usar PyTorch.")
        if hardware["gpu_devices"]:
            gpu_fraction = st.slider("Límite del asignador de memoria CUDA (%)", 40, 95,
                                     value=round(defaults["gpu_memory_fraction"] * 100), step=5,
                                     help="Límite para el asignador CUDA de PyTorch, no un límite físico garantizado para toda la GPU.")
        else:
            gpu_fraction = round(defaults["gpu_memory_fraction"] * 100)
        st.markdown("#### Tamaño y duración del entrenamiento")
        a, b, c = st.columns(3)
        dimensions = a.number_input("Dimensiones del embedding", 8, 512, int(defaults["dimensions"]), step=8)
        epochs = b.number_input("Épocas", 1, 200, int(defaults["epochs"]))
        batch_size = c.number_input("Tamaño del lote", 16, 4096, int(defaults["batch_size"]), step=16)
        d, e, f = st.columns(3)
        window = d.number_input("Ventana de contexto", 1, 10, int(defaults["window"]))
        negatives = e.number_input("Muestras negativas", 1, 20, int(defaults["negatives"]))
        max_pairs = f.number_input("Máximo de pares de entrenamiento", 100, 1000000,
                                    int(defaults["max_pairs"]), step=1000)
        with st.expander("Opciones avanzadas", expanded=False):
            learning_rate = st.number_input("Tasa de aprendizaje", 0.00001, 0.1,
                                             float(defaults["learning_rate"]), step=0.0005, format="%.5f")
            seed = st.number_input("Semilla aleatoria", 0, 2147483647, int(defaults["seed"]))
            snapshot_every = st.number_input("Guardar embeddings cada N épocas", 1, 200,
                                               int(defaults["snapshot_every"]))
        st.caption("Lotes, dimensiones y muestras negativas mayores pueden aumentar el consumo "
                   "de memoria. Más épocas y pares aumentan principalmente la duración del entrenamiento.")
        submitted = st.form_submit_button(
            "Guardar y continuar" if first_time else "Guardar configuración", type="primary")
    if submitted:
        selected = {
            "profile": profile, "device": device, "cpu_threads": int(cpu_threads),
            "gpu_memory_fraction": float(gpu_fraction) / 100,
            "dimensions": int(dimensions), "epochs": int(epochs),
            "batch_size": int(batch_size), "window": int(window),
            "negatives": int(negatives), "max_pairs": int(max_pairs),
            "learning_rate": float(learning_rate), "seed": int(seed),
            "snapshot_every": int(snapshot_every),
        }
        try:
            save_settings(settings_path, selected, hardware)
        except ValueError as exc:
            st.error(f"No se pudo guardar: {exc}")
        else:
            st.session_state["config_saved_notice"] = True
            st.rerun()
    if st.session_state.pop("config_saved_notice", False):
        st.success("Configuración guardada correctamente para esta computadora.")
