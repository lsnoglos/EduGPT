"""Interfaz local del primer hito EduGPT. Sin llamadas a IA ni modelos preentrenados."""
from datetime import datetime
from io import BytesIO
from pathlib import Path
import json
import re
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import streamlit as st
import torch

from edugpt.core import SUPPORTED, prepare_corpus
from edugpt.trainer import neighbors, train
from edugpt.hardware import detect_hardware, load_settings, device_options
from edugpt.settings_ui import settings_editor

ROOT = Path(__file__).resolve().parent
BOOKS = ROOT / "data" / "books"
CORPUS = ROOT / "data" / "processed" / "corpus.json"
RUNS = ROOT / "artifacts" / "runs"
SETTINGS = ROOT / "data" / "settings.json"
BOOKS.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)
st.set_page_config(page_title="EduGPT | Laboratorio IA", page_icon="🧠", layout="wide")
st.title("🧠 EduGPT · Laboratorio de aprendizaje")
st.caption("Una IA educativa entrenada desde cero con libros locales. Sin APIs ni pesos preentrenados.")
hardware = detect_hardware()
if not SETTINGS.exists():
    settings_editor(SETTINGS, hardware, first_time=True)
    st.stop()

settings = load_settings(SETTINGS, hardware)
if settings["device"] == "auto":
    resolved_name = "GPU CUDA" if hardware["gpu_devices"] else "CPU"
else:
    resolved_name = settings["device"].upper()
st.caption(f'Configuración actual: {settings["profile"]} · Dispositivo: {resolved_name} · '
           f'{settings["cpu_threads"]} hilos CPU · Lote {settings["batch_size"]}')

# Si una GPU explícitamente elegida desaparece, exigir que se revise la selección.
try:
    stored_device = json.loads(SETTINGS.read_text(encoding="utf-8")).get("device", "auto")
except (OSError, ValueError):
    stored_device = "auto"
needs_device_review = stored_device not in device_options(hardware)
if needs_device_review:
    st.error(f"La selección guardada ({stored_device}) ya no está disponible. "
             "Abre Configuración y guarda un dispositivo válido antes de entrenar.")

books_tab, training_tab, explorer_tab, experiment_tab, config_tab = st.tabs([
    "📚 Biblioteca", "⚙️ Entrenamiento", "🔎 Explorar palabras",
    "📊 Experimentos", "🛠️ Configuración"])

with books_tab:
    st.subheader("Libros que alimentan el modelo")
    uploads = st.file_uploader("Cargar archivos TXT, PDF o DOCX", type=["txt", "pdf", "docx"], accept_multiple_files=True)
    if st.button("Guardar libros", disabled=not uploads):
        for upload in uploads:
            safe_name = re.sub(r"[^\w.\-]", "_", Path(upload.name).name)
            if Path(safe_name).suffix.lower() not in SUPPORTED:
                continue
            if upload.size > 25 * 1024 * 1024:
                st.warning(f"{safe_name}: supera el límite de 25 MB por archivo.")
                continue
            (BOOKS / safe_name).write_bytes(upload.getvalue())
        st.success("Archivos guardados en la biblioteca local.")
    st.write("**Archivos disponibles:**", ", ".join(p.name for p in BOOKS.iterdir() if p.suffix.lower() in SUPPORTED) or "ninguno")
    minimum = st.number_input("Frecuencia mínima de palabras", min_value=1, max_value=20, value=2)
    vocab_limit = st.number_input("Vocabulario máximo", min_value=100, max_value=50000, value=8000, step=100)
    if st.button("Preparar corpus y vocabulario", type="primary"):
        try:
            result = prepare_corpus(BOOKS, CORPUS, int(minimum), int(vocab_limit))
            st.success("Corpus preparado. Ya puedes entrenar los embeddings.")
            st.json(result["stats"])
        except Exception as exc:
            st.error(f"No se pudo preparar el corpus: {exc}")
    if CORPUS.exists():
        st.caption("Corpus existente (actualízalo si agregaste libros):")
        st.json(json.loads(CORPUS.read_text(encoding="utf-8"))["stats"])

with training_tab:
    st.subheader("Skip-gram · Entrenamiento controlado")
    st.write("El entrenamiento utiliza exclusivamente los libros locales y los recursos configurados.")
    st.info("Puedes cambiar CPU/GPU, hilos, memoria y todos los hiperparámetros en **🛠️ Configuración**. "
            "Los cambios se aplicarán en el siguiente entrenamiento.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dispositivo", resolved_name)
    c2.metric("Épocas", settings["epochs"])
    c3.metric("Lote", settings["batch_size"])
    c4.metric("Dimensiones", settings["dimensions"])
    with st.expander("Ver todos los parámetros antes de entrenar"):
        st.json(settings)
    if st.button("Entrenar embeddings con estos valores", type="primary",
                 disabled=not CORPUS.exists() or needs_device_review):
        corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
        run_dir = RUNS / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        progress = st.progress(0, text="Inicializando...")
        placeholder = st.empty()
        epochs = settings["epochs"]

        def update(row, history):
            progress.progress(row["epoch"] / epochs, text=f"Época {row['epoch']}/{epochs} · pérdida {row['loss']:.4f}")
            placeholder.line_chart(pd.DataFrame(history).set_index("epoch")["loss"])

        try:
            train(corpus, run_dir, dimensions=settings["dimensions"],
                  window=settings["window"], epochs=epochs,
                  negatives=settings["negatives"], batch_size=settings["batch_size"],
                  max_pairs=settings["max_pairs"], learning_rate=settings["learning_rate"],
                  seed=settings["seed"], device=settings["device"],
                  cpu_threads=settings["cpu_threads"],
                  gpu_memory_fraction=settings["gpu_memory_fraction"],
                  snapshot_every=settings["snapshot_every"], on_epoch=update)
            st.success(f"Entrenamiento finalizado. Experimento: {run_dir.name}")
            st.info("Visita 'Explorar palabras' y 'Experimentos' para revisar los resultados reales.")
        except torch.cuda.OutOfMemoryError:
            st.error("Memoria GPU insuficiente: reduce lote, dimensiones o muestras negativas "
                     "en Configuración; también puedes elegir CPU.")
        except (RuntimeError, ValueError, OSError) as exc:
            st.error(f"El entrenamiento no se completó: {exc}")

run_dirs = sorted((p for p in RUNS.iterdir() if (p / "embeddings_final.npy").exists()), reverse=True)
with st.sidebar:
    st.header("Experimento activo")
    run_name = st.selectbox("Selecciona una ejecución", [p.name for p in run_dirs]) if run_dirs else None
run_dir = RUNS / run_name if run_name else None

with explorer_tab:
    if run_dir is None:
        st.info("Entrena primero un modelo para explorar sus palabras.")
    else:
        vocab = json.loads((run_dir / "vocab.json").read_text(encoding="utf-8"))
        final = np.load(run_dir / "embeddings_final.npy", allow_pickle=False)
        initial = np.load(run_dir / "embeddings_initial.npy", allow_pickle=False)
        word = st.selectbox("Palabra", sorted(w for w in vocab if w != "<unk>"))
        st.write("**Vecinos más cercanos (similitud coseno)**")
        st.dataframe(pd.DataFrame(neighbors(final, vocab, word, k=10)), hide_index=True, use_container_width=True)
        st.write("**Desplazamiento del vector entre el inicio y el final**")
        idx = vocab[word]
        st.metric("Distancia euclidiana", f"{np.linalg.norm(final[idx] - initial[idx]):.5f}")
        st.caption("Los vecinos describen asociaciones estadísticas, no conocimiento semántico garantizado.")
        st.subheader("Mapa PCA de embeddings")
        limit = st.slider("Palabras visibles", min_value=10, max_value=min(500, len(vocab)-1),
                          value=min(150, len(vocab)-1)) if len(vocab) > 11 else len(vocab)-1
        terms = list(vocab.keys())[1:limit+1]
        ids = [vocab[w] for w in terms]
        # PCA ajustada conjuntamente para comparar coordenadas en una misma base.
        both = np.concatenate([initial[ids], final[ids]], axis=0)
        coords = PCA(n_components=2).fit_transform(both)
        final_coords = coords[len(ids):]
        clusters = KMeans(n_clusters=min(5, len(ids)), random_state=42, n_init=10).fit_predict(final_coords)
        plot_data = pd.DataFrame({"x": final_coords[:, 0], "y": final_coords[:, 1],
                                  "palabra": terms, "grupo": clusters.astype(str)})
        fig = px.scatter(plot_data, x="x", y="y", color="grupo", hover_name="palabra",
                         title="Proyección PCA · agrupaciones K-means")
        fig.update_traces(marker={"size": 9})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Los grupos son estadísticos y la distancia en 2D no representa toda la geometría original.")

with experiment_tab:
    if run_dir is None:
        st.info("Aún no hay entrenamientos finalizados.")
    else:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        metrics = pd.read_csv(run_dir / "metrics.csv")
        st.json(config)
        st.line_chart(metrics.set_index("epoch")[["loss", "gradient_norm"]])
        st.line_chart(metrics.set_index("epoch")["pair_accuracy"])
        st.caption("La métrica 'pair_accuracy' mide discriminación entre pares positivos y negativos; no equivale a precisión de generación de texto.")
        data = BytesIO()
        with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in run_dir.iterdir():
                if file.is_file():
                    archive.write(file, arcname=file.name)
        st.download_button("Exportar experimento (ZIP)", data=data.getvalue(),
                           file_name=f"EduGPT_{run_name}.zip", mime="application/zip")

with config_tab:
    settings_editor(SETTINGS, hardware)
