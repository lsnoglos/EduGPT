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

ROOT = Path(__file__).resolve().parent
BOOKS = ROOT / "data" / "books"
CORPUS = ROOT / "data" / "processed" / "corpus.json"
RUNS = ROOT / "artifacts" / "runs"
BOOKS.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)
st.set_page_config(page_title="EduGPT | Laboratorio IA", page_icon="🧠", layout="wide")
st.title("🧠 EduGPT · Laboratorio de aprendizaje")
st.caption("Una IA educativa entrenada desde cero con libros locales. Sin APIs ni pesos preentrenados.")
if torch.cuda.is_available():
    st.success(f"GPU CUDA disponible: {torch.cuda.get_device_name(0)}")
else:
    st.info("Modo CPU activo. En la laptop con RTX 3070, instala PyTorch con CUDA para acelerar el entrenamiento.")

books_tab, training_tab, explorer_tab, experiment_tab = st.tabs([
    "📚 Biblioteca", "⚙️ Entrenamiento", "🔎 Explorar palabras", "📊 Experimentos"])

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
    st.subheader("Skip-gram + muestreo negativo")
    st.write("Pesos inicializados aleatoriamente; cada métrica procede de una ejecución real de PyTorch.")
    c1, c2, c3 = st.columns(3)
    dimensions = c1.selectbox("Dimensiones de embeddings", [32, 64, 128, 256], index=1)
    window = c2.slider("Ventana de contexto", 1, 6, 2)
    epochs = c3.slider("Épocas", 1, 50, 10)
    c4, c5, c6 = st.columns(3)
    negatives = c4.slider("Ejemplos negativos", 1, 15, 5)
    batch = c5.selectbox("Lote", [128, 256, 512, 1024], index=2)
    max_pairs = c6.number_input("Máximo de pares", min_value=1000, max_value=1000000, value=200000, step=1000)
    if st.button("Entrenar embeddings", type="primary", disabled=not CORPUS.exists()):
        corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
        run_dir = RUNS / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        progress = st.progress(0, text="Inicializando...")
        placeholder = st.empty()

        def update(row, history):
            progress.progress(row["epoch"] / epochs, text=f"Época {row['epoch']}/{epochs} · pérdida {row['loss']:.4f}")
            placeholder.line_chart(pd.DataFrame(history).set_index("epoch")["loss"])

        try:
            train(corpus, run_dir, dimensions=dimensions, window=window,
                  epochs=epochs, negatives=negatives, batch_size=batch,
                  max_pairs=int(max_pairs), on_epoch=update)
            st.success(f"Entrenamiento finalizado. Experimento: {run_dir.name}")
            st.info("Visita las pestañas 'Explorar palabras' y 'Experimentos' para revisar el resultado.")
        except Exception as exc:
            st.error(f"Error de entrenamiento: {exc}")

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
