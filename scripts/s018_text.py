"""Retraining recipe text pipeline: misaki G2P (American English, espeak fallback) -> Kokoro vocab ids.

Same text frontend as the teacher package `kokoro` (KPipeline lang 'a' uses
misaki en.G2P with the espeak fallback). ids = [0] + vocab ids + [0]; phoneme
characters outside the vocab are dropped, as KModel does. The vocab comes from
the teacher's config.json (models/pytorch/config.json by default); call
set_vocab(path) to use another file.
"""

import json
import re
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOCAB = ROOT / "models/pytorch/config.json"
VOCAB: dict[str, int] = {}
ID_TO_TOKEN: dict[int, str] = {}

_G2P = None


def set_vocab(path: Path | None = None) -> None:
    """Load the phoneme -> id table in place (module-level dicts keep their identity)."""
    vocab = json.loads(Path(path or DEFAULT_VOCAB).read_text())["vocab"]
    VOCAB.clear(); VOCAB.update(vocab)
    ID_TO_TOKEN.clear(); ID_TO_TOKEN.update({i: t for t, i in vocab.items()})


def g2p():
    global _G2P
    if _G2P is None:
        warnings.filterwarnings("ignore")
        from misaki import en, espeak

        _G2P = en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False))
    return _G2P


def phonemize(text: str) -> tuple[str, list[int]]:
    """text -> (phoneme string, ids WITHOUT the two pads)."""
    if not VOCAB:
        set_vocab()
    phonemes, _ = g2p()(text)
    return phonemes, [VOCAB[c] for c in phonemes if c in VOCAB]


def normalize(text: str) -> str:
    """Lowercase, apostrophes unified, punctuation removed, spaces collapsed."""
    text = text.lower().replace("’", "'")
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", text).split())


def read_sentences(paths, min_chars: int = 1, max_chars: int = 10**9, max_words: int = 10**9) -> list[str]:
    """Split plain-text files into single sentences (paragraphs on blank lines, then ., ! or ?)."""
    split = re.compile(r"(?<=[.!?])[\"”’)]*\s+(?=[\"“‘(]?[A-Z])")
    out = []
    for path in paths:
        for para in re.split(r"\n\s*\n", Path(path).read_text(encoding="utf-8")):
            para = " ".join(para.split())
            for s in split.split(para):
                s = s.strip()
                if min_chars <= len(s) <= max_chars and len(s.split()) <= max_words:
                    out.append(s)
    return out
