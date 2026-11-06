"""Preparación reproducible de un corpus compuesto únicamente por libros locales."""
from collections import Counter
from pathlib import Path
import json
import re
import unicodedata

SUPPORTED = {".txt", ".pdf", ".docx"}
TOKEN_PATTERN = re.compile(r"[a-záéíóúüñ]+(?:[’'-][a-záéíóúüñ]+)?|\d+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).lower()
    return TOKEN_PATTERN.findall(text)


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8-sig")
    if suffix == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    if suffix == ".docx":
        from docx import Document
        return "\n".join(p.text for p in Document(str(path)).paragraphs)
    raise ValueError(f"Formato no admitido: {suffix}")


def prepare_corpus(books_dir: Path, output: Path, min_freq: int = 2,
                   max_vocab: int = 8000) -> dict:
    books = sorted(p for p in books_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not books:
        raise ValueError("Primero agrega libros TXT, PDF o DOCX a la biblioteca.")
    documents = []
    for book in books:
        words = tokenize(extract_text(book))
        if words:
            documents.append({"book": book.name, "tokens": words})
    if not documents:
        raise ValueError("Los archivos no contienen texto extraíble; un PDF escaneado requiere OCR.")
    counts = Counter(w for d in documents for w in d["tokens"])
    ordered = sorted((w for w, n in counts.items() if n >= min_freq),
                     key=lambda w: (-counts[w], w))[:max_vocab - 1]
    if len(ordered) < 2:
        raise ValueError("Se necesitan al menos dos palabras distintas; baja la frecuencia mínima.")
    vocab = {"<unk>": 0, **{w: i + 1 for i, w in enumerate(ordered)}}
    result = {
        "vocab": vocab,
        "frequencies": [0] + [counts[w] for w in ordered],
        "documents": [{"book": d["book"], "ids": [vocab.get(w, 0) for w in d["tokens"]]}
                      for d in documents],
        "stats": {"books": len(documents), "tokens": sum(map(lambda d: len(d["tokens"]), documents)),
                  "vocab_size": len(vocab), "unknown_tokens": sum(counts[w] for w in counts if w not in vocab)}
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result
