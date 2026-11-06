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
3. En el **primer inicio**, EduGPT detecta núcleos CPU, RAM, GPUs CUDA y VRAM; acepta el perfil Equilibrado recomendado o ajusta las opciones y guarda la configuración local.\n4. En **🛠️ Configuración**, selecciona **Automático, CPU o una GPU CUDA concreta** y personaliza los perfiles Ahorrador, Equilibrado o Alto rendimiento, núcleos/hilos CPU, cuota del asignador CUDA, lote, dimensiones, épocas, negativos, ventana, máximo de pares, tasa de aprendizaje, semilla y frecuencia de checkpoints.\n5. En **Entrenamiento**, comprueba el resumen de recursos y pulsa **Entrenar embeddings**.
6. Examina pérdida, gradientes, RAM y memoria GPU en **Experimentos** y similitud coseno y mapa PCA en **Explorar palabras**.
7. Exporta el ZIP del experimento con pesos, vocabulario, parámetros efectivos, métricas y vectores.

Cada entrenamiento comienza con pesos inicializados aleatoriamente y se guarda en `artifacts/runs/`. Solo usa CUDA si se selecciona expresamente o si el dispositivo es **Automático** y hay CUDA. Elegir **CPU** fuerza el entrenamiento en CPU aunque exista una GPU. Elegir una GPU que no esté disponible genera un error en lugar de cambiar silenciosamente a CPU.\n\nLas preferencias se guardan en `data/settings.json` (no se sube a GitHub). El límite de VRAM se aplica al **asignador de PyTorch**, no a toda la memoria consumida por el proceso ni a otras aplicaciones. Los perfiles son puntos de partida, no predicciones de tiempo ni garantías de ausencia de OOM. Al cambiar de equipo, la selección de GPU que no exista debe revisarse antes de entrenar.

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
edugpt/trainer.py      Skip-gram, CPU/CUDA, métricas y checkpoints\nedugpt/hardware.py     Detección de hardware y perfiles de consumo\nedugpt/settings_ui.py  Asistente inicial y formulario de configuración\ndata/settings.json     Preferencias locales, no se sincronizan por Git
tests/                 Pruebas automatizadas
data/books/            Libros locales (ignorado por Git)
artifacts/runs/        Modelos y métricas locales (ignorado por Git)
```

**Principio central:** los libros aportan el texto; PyTorch, NumPy, PCA y K-means son herramientas matemáticas, no fuentes de conocimiento lingüístico preentrenado.
