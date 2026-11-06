"""Skip-gram con muestreo negativo: pesos aleatorios y entrenamiento real en PyTorch."""
from pathlib import Path
import csv
import json
import random
import time

import numpy as np
import psutil
import torch

from edugpt.hardware import resolve_device
from torch import nn
from torch.nn import functional as F


class SkipGramNS(nn.Module):
    def __init__(self, vocab_size: int, dimensions: int):
        super().__init__()
        self.input = nn.Embedding(vocab_size, dimensions)
        self.output = nn.Embedding(vocab_size, dimensions)
        nn.init.uniform_(self.input.weight, -0.5 / dimensions, 0.5 / dimensions)
        nn.init.zeros_(self.output.weight)

    def forward(self, centers, positive, negative):
        center = self.input(centers)
        pos_score = (center * self.output(positive)).sum(dim=1)
        neg_score = torch.bmm(self.output(negative), center.unsqueeze(2)).squeeze(2)
        loss = F.softplus(-pos_score) + F.softplus(neg_score).sum(dim=1)
        accuracy = ((pos_score > 0).float().sum() + (neg_score < 0).float().sum()) / (
            pos_score.numel() + neg_score.numel())
        return loss.mean(), accuracy.item()


def make_pairs(documents: list[dict], window: int, max_pairs: int, seed: int) -> np.ndarray:
    """Nunca forma pares entre libros distintos; el 0 (<unk>) se omite."""
    rng = random.Random(seed)
    documents = list(documents)
    rng.shuffle(documents)
    pairs = []
    for document in documents:
        ids = document["ids"]
        positions = list(range(len(ids)))
        rng.shuffle(positions)
        for i in positions:
            if ids[i] == 0:
                continue
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if i != j and ids[j] != 0:
                    pairs.append((ids[i], ids[j]))
                    if len(pairs) >= max_pairs:
                        return np.asarray(pairs, dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def train(corpus: dict, run_dir: Path, *, dimensions: int = 64, window: int = 2,
          epochs: int = 10, negatives: int = 5, batch_size: int = 512,
          max_pairs: int = 200000, learning_rate: float = 0.003,
          seed: int = 42, on_epoch=None, device: str = "auto",
          cpu_threads: int | None = None, gpu_memory_fraction: float = 0.80,
          snapshot_every: int = 5) -> list[dict]:
    if any(x < 1 for x in (dimensions, window, epochs, negatives, batch_size, max_pairs)):
        raise ValueError("Los hiperparámetros enteros deben ser positivos.")
    if not 0.4 <= gpu_memory_fraction <= 0.95:
        raise ValueError("El límite de VRAM debe estar entre 40 % y 95 %.")
    if snapshot_every < 1:
        raise ValueError("snapshot_every debe ser positivo.")
    if cpu_threads is not None:
        if cpu_threads < 1:
            raise ValueError("cpu_threads debe ser positivo.")
        torch.set_num_threads(cpu_threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    pairs = make_pairs(corpus["documents"], window, max_pairs, seed)
    if len(pairs) < 2:
        raise ValueError("No hay suficientes pares de palabras para entrenar.")
    requested_device = device
    device = resolve_device(requested_device)
    if device.type == "cuda":
        # Limita solo el allocador CUDA de PyTorch, no otras aplicaciones o bibliotecas.
        torch.cuda.set_per_process_memory_fraction(float(gpu_memory_fraction), device=device)
        torch.cuda.reset_peak_memory_stats(device)
    model = SkipGramNS(len(corpus["vocab"]), dimensions).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)
    frequencies = torch.tensor(corpus["frequencies"], dtype=torch.float32, device=device)
    distribution = frequencies.pow(0.75)
    distribution[0] = 0
    distribution /= distribution.sum()
    run_dir.mkdir(parents=True, exist_ok=False)
    np.save(run_dir / "embeddings_initial.npy", model.input.weight.detach().cpu().numpy())
    (run_dir / "vocab.json").write_text(json.dumps(corpus["vocab"], ensure_ascii=False), encoding="utf-8")
    config = {"dimensions": dimensions, "window": window, "epochs": epochs,
              "negatives": negatives, "batch_size": batch_size, "max_pairs": max_pairs,
              "learning_rate": learning_rate, "seed": seed, "device": str(device),
              "requested_device": requested_device, "cpu_threads": torch.get_num_threads(),
              "gpu_memory_fraction": gpu_memory_fraction if device.type == "cuda" else None,
              "snapshot_every": snapshot_every,
              "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "training_pairs": len(pairs), "books": [d["book"] for d in corpus["documents"]]}
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    records = []
    data = torch.from_numpy(pairs)
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(data))
        sum_loss = sum_accuracy = sum_grad = 0.0
        count = 0
        for offset in range(0, len(data), batch_size):
            chunk = data[permutation[offset:offset + batch_size]].to(device)
            negatives_ids = torch.multinomial(distribution, chunk.shape[0] * negatives,
                                             replacement=True).reshape(-1, negatives)
            optimizer.zero_grad(set_to_none=True)
            loss, accuracy = model(chunk[:, 0], chunk[:, 1], negatives_ids)
            loss.backward()
            grad = model.input.weight.grad.detach().norm().item()
            optimizer.step()
            n = len(chunk)
            sum_loss += loss.item() * n
            sum_accuracy += accuracy * n
            sum_grad += grad * n
            count += n
        ram = psutil.virtual_memory()
        row = {"epoch": epoch, "loss": sum_loss / count,
               "pair_accuracy": sum_accuracy / count,
               "gradient_norm": sum_grad / count, "elapsed_seconds": round(time.perf_counter() - start, 2),
               "cpu_process_ram_gb": round(psutil.Process().memory_info().rss / (1024 ** 3), 3),
               "ram_available_gb": round(ram.available / (1024 ** 3), 3),
               "gpu_peak_allocated_gb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 3)
                   if device.type == "cuda" else 0.0,
               "gpu_reserved_gb": round(torch.cuda.memory_reserved(device) / (1024 ** 3), 3)
                   if device.type == "cuda" else 0.0}
        records.append(row)
        if epoch == 1 or epoch == epochs or epoch % snapshot_every == 0:
            np.save(run_dir / f"embeddings_epoch_{epoch:03d}.npy",
                    model.input.weight.detach().cpu().numpy())
            torch.save(model.state_dict(), run_dir / "model.pt")
        with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(records)
        if on_epoch is not None:
            on_epoch(row, records)
    np.save(run_dir / "embeddings_final.npy", model.input.weight.detach().cpu().numpy())
    return records


def neighbors(embedding: np.ndarray, vocab: dict, word: str, k: int = 5) -> list[dict]:
    if word not in vocab or vocab[word] == 0:
        return []
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    normalized = embedding / np.maximum(norms, 1e-12)
    scores = normalized @ normalized[vocab[word]]
    scores[0] = -np.inf
    scores[vocab[word]] = -np.inf
    inverse = {index: token for token, index in vocab.items()}
    return [{"palabra": inverse[int(i)], "similitud": round(float(scores[i]), 5)}
            for i in np.argsort(-scores)[:k] if np.isfinite(scores[i])]
