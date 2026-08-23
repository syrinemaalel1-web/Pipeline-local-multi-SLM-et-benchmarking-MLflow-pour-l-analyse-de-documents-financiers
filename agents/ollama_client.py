"""Accès à Ollama : appel chat instrumenté et mesure mémoire.

Deux endpoints sont utilisés, chacun pour ce qu'il sait faire :

- l'endpoint compatible OpenAI (`/v1`) pour les complétions, parce que
  `mlflow.openai.autolog()` le trace sans code de tracing manuel ;
- l'API native (`/api/ps`, `/api/generate`) pour ce que la couche OpenAI n'expose
  pas : l'occupation VRAM réelle et le préchargement du modèle.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

import requests

from config import (
    JUDGE_BASE_URL,
    MAX_TOKENS_BY_TASK,
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    REASONING_MODELS,
    REQUEST_TIMEOUT_S,
    SEED,
    TEMPERATURE_BY_TASK,
    Task,
)

log = logging.getLogger(__name__)

_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


@dataclass
class MemorySample:
    """Occupation mémoire d'un modèle chargé, telle que rapportée par Ollama."""

    total_bytes: int
    vram_bytes: int

    @property
    def gpu_fraction(self) -> float:
        """Part du modèle réellement sur GPU.

        Sur une carte de 4 Go face à des modèles de 5 Go, cette fraction est le vrai
        discriminant de coût : la taille totale est presque identique d'un candidat
        à l'autre.
        """
        return self.vram_bytes / self.total_bytes if self.total_bytes else 0.0


@dataclass
class AgentCall:
    """Résultat d'un appel agent, avec tout ce que le rapport devra recouper."""

    model: str
    task: str
    output: str
    raw_output: str
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    memory: dict[str, Any] | None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float | None:
        if not self.completion_tokens or self.latency_s <= 0:
            return None
        return self.completion_tokens / self.latency_s

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "tokens_per_second": self.tokens_per_second}


@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI

    return OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key=OLLAMA_API_KEY,
        timeout=REQUEST_TIMEOUT_S,
        max_retries=0,
    )


def read_memory(model: str) -> MemorySample | None:
    """Lit l'occupation mémoire d'un modèle chargé via `/api/ps`.

    Renvoie ``None`` si le modèle n'est pas chargé. `/api/ps` est préféré à la sortie
    texte de `ollama ps` : il donne `size_vram` en octets, sans parsing fragile.
    """
    try:
        resp = requests.get(f"{JUDGE_BASE_URL}/api/ps", timeout=10)
        resp.raise_for_status()
        entries = resp.json().get("models", [])
    except Exception as exc:
        log.debug("Lecture de /api/ps impossible : %s", exc)
        return None

    for entry in entries:
        if entry.get("name") == model or entry.get("model") == model:
            return MemorySample(
                total_bytes=int(entry.get("size", 0)),
                vram_bytes=int(entry.get("size_vram", 0)),
            )
    return None


class _MemoryProbe(threading.Thread):
    """Échantillonne la mémoire pendant la génération.

    Nécessaire parce qu'`/api/ps` ne liste un modèle que tant qu'il est chargé :
    interroger après coup peut tomber sur une table vide.
    """

    def __init__(self, model: str, interval_s: float = 2.0) -> None:
        super().__init__(daemon=True)
        self.model = model
        self.interval_s = interval_s
        self.peak: MemorySample | None = None
        # Surtout pas `_stop` : c'est une méthode interne de threading.Thread,
        # la masquer fait échouer la terminaison du thread.
        self._done = threading.Event()

    def run(self) -> None:
        while not self._done.is_set():
            sample = read_memory(self.model)
            if sample and (
                self.peak is None or sample.total_bytes > self.peak.total_bytes
            ):
                self.peak = sample
            self._done.wait(self.interval_s)

    def stop(self) -> MemorySample | None:
        self._done.set()
        self.join(timeout=5)
        return self.peak


def warm_model(model: str) -> bool:
    """Précharge un modèle et fixe son `keep_alive`.

    Deux raisons de le faire explicitement : la couche compatible OpenAI n'expose pas
    `keep_alive`, et surtout le temps de chargement (5 Go depuis le disque) doit
    rester hors des latences mesurées, sinon le premier document de chaque modèle
    porte seul le coût du chargement.
    """
    try:
        resp = requests.post(
            f"{JUDGE_BASE_URL}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": OLLAMA_KEEP_ALIVE},
            timeout=REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
        return True
    except Exception as exc:
        log.error("Préchargement de %s impossible : %s", model, exc)
        return False


def clean_output(text: str, model: str) -> str:
    """Retire le bloc de raisonnement des modèles qui en émettent un.

    `deepseek-r1:8b` raisonne souvent en anglais même sur un document arabe : laisser
    le bloc `<think>` polluerait à la fois le juge de fidélité et le scorer de langue.
    """
    if model in REASONING_MODELS:
        text = _THINK.sub("", text)
        # Un bloc tronqué par max_tokens laisse une balise ouvrante orpheline.
        if "<think>" in text and "</think>" not in text:
            text = text.split("<think>", 1)[0]
    return text.strip()


def chat(
    model: str,
    messages: list[dict[str, str]],
    task: Task,
    *,
    probe_memory: bool = True,
) -> AgentCall:
    """Appelle Ollama et renvoie la sortie nettoyée, la latence et la mémoire.

    Les paramètres d'inférence viennent tous de `config` : température par tâche,
    seed fixe, plafond de génération. Aucun n'est ajustable par modèle, sans quoi la
    comparaison ne voudrait plus rien dire.
    """
    probe = _MemoryProbe(model) if probe_memory else None
    if probe:
        probe.start()

    started = time.perf_counter()
    error: str | None = None
    raw_output = ""
    usage = None

    try:
        response = _openai_client().chat.completions.create(
            model=model,
            messages=messages,
            temperature=TEMPERATURE_BY_TASK[task],
            max_tokens=MAX_TOKENS_BY_TASK[task],
            seed=SEED,
        )
        raw_output = response.choices[0].message.content or ""
        usage = response.usage
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        log.error("Appel %s / %s en échec : %s", model, task.value, error)

    latency_s = time.perf_counter() - started
    peak = probe.stop() if probe else None

    return AgentCall(
        model=model,
        task=task.value,
        output=clean_output(raw_output, model),
        raw_output=raw_output,
        latency_s=round(latency_s, 3),
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        memory=(
            {
                "total_bytes": peak.total_bytes,
                "vram_bytes": peak.vram_bytes,
                "gpu_fraction": round(peak.gpu_fraction, 4),
            }
            if peak
            else None
        ),
        error=error,
    )
