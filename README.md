# EduGPT — Laboratorio de IA desde cero

Primer hito funcional: **Skip-gram con muestreo negativo**, sin modelos preentrenados, APIs de IA ni datos de entrenamiento ajenos a los libros que cargue el usuario.

## Requisitos
- Python 3.11 recomendado; Python 3.10 o superior.
- 16 GB de RAM y, opcionalmente, NVIDIA RTX 3070 8 GB (CUDA).
- El funcionamiento posterior a la instalación puede ser completamente offline.

## Instalación en Windows PowerShell
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Para utilizar la RTX 3070 instala **PyTorch con CUDA**, siguiendo el selector oficial: https://pytorch.org/get-started/locally/
Después verifica que se detecte la GPU:
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Ejecutar
```powershell
streamlit run app.py
```

1. En **Biblioteca**, carga libros TXT, PDF con texto seleccionable o DOCX y pulsa **Guardar libros**.
2. Pulsa **Preparar corpus y vocabulario**. El resultado se guarda en `data/processed/`.
3. En **Entrenamiento**, ajusta los hiperparámetros y pulsa **Entrenar embeddings**.
4. Examina pérdida y gradientes en **Experimentos** y la similitud coseno y el mapa PCA en **Explorar palabras**.
5. Exporta el ZIP del experimento con los pesos, vocabulario, configuración, métricas y vectores.

Cada entrenamiento comienza con pesos inicializados aleatoriamente, se guarda en `artifacts/runs/` y utiliza CUDA si está disponible.

## Interpretación científica
- Los resultados y métricas son reales. `pair_accuracy` mide discriminación de pares positivos y negativos, **no** precisión de generación ni comprensión lingüística.
- Las coordenadas PCA y los grupos K-means son patrones estadísticos, no categorías semánticas garantizadas.
- Las matrices de embeddings se conservan en el estado inicial, final y algunas épocas intermedias.
- Los PDF escaneados necesitan OCR y no están admitidos en este hito.
- Aún **no** incluye Transformer, generación de texto, búsqueda de citas en libros ni importación ZIP. Son próximas fases.
- No publiques libros protegidos por derechos de autor sin autorización.

## Pruebas
```powershell
python -m pip install pytest
python -m pytest -q
```

## Estructura
```text
app.py                 Interfaz web Streamlit
edugpt/core.py         Corpus, tokens, vocabulario e IDs
edugpt/trainer.py      Skip-gram, métricas y checkpoints
tests/                 Pruebas automatizadas
data/books/            Libros locales (ignorado por Git)
artifacts/runs/        Modelos y métricas locales (ignorado por Git)
```

**Principio central:** los libros aportan el texto; PyTorch, NumPy, PCA y K-means son herramientas matemáticas, no fuentes de conocimiento lingüístico preentrenado.
