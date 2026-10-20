"""Build S014 distillation text from public-domain Gutenberg books.

Run with the kokoro-onnx venv and -I (the books are downloaded data):

    .venv-convert/bin/python -I scripts/s014_corpus.py

    books -> strip Gutenberg header/footer -> paragraphs -> sentences
    -> group neighbouring sentences to a random target length
    -> espeak-ng en-us phonemes, exactly as the frozen chunks were made
    -> token ids; keep 3..508 tokens so pads fit Kokoro's 510 limit

Train and holdout use different books by different authors.
"""

import json
import random
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "artifacts/s014/corpus"
RAW = CORPUS / "raw"
MANIFEST = ROOT / "benchmark/inputs/manifest.json"

TRAIN_BOOKS = (
    1342, 11, 84, 98, 1661, 2701, 345, 174, 76, 74, 1260, 768, 120, 219, 43,
    1400, 158, 161, 205, 2554, 36, 35, 5230, 2591, 64317, 1232, 2199, 1727,
    2009, 1497, 4363, 147, 46, 25344, 514, 2814, 55, 16, 244, 113, 45, 23,
    1250, 2852,
)
# London, Shaw, Kafka, Hesse, Franklin: no author overlap with training.
HOLDOUT_BOOKS = (215, 3825, 5200, 2500, 20203)
PER_TRAIN_BOOK = 2_000
PER_HOLDOUT_BOOK = 300

SENTENCE_END = re.compile(r"(?<=[.!?])[\"”’)]*\s+(?=[\"“‘(]?[A-Z])")
ABBREVIATION = re.compile(r"\b(Mr|Mrs|Ms|Dr|St|Jr|Sr|Mt|Capt|Col|Gen|Rev|Prof|No|vs|etc|[A-Z])\.$")


def paragraphs(book: int) -> list[str]:
    text = (RAW / f"pg{book}.txt").read_text(encoding="utf-8", errors="ignore")
    start = re.search(r"\*\*\* ?START OF.*?\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF", text)
    text = text[start.end() if start else 0 : end.start() if end else len(text)]
    text = re.sub(r"\[[^\]]*\]", " ", text)  # [Illustration], footnote marks
    text = text.replace("_", "").replace("\r", "")
    result = []
    for block in re.split(r"\n\s*\n", text):
        block = " ".join(block.split())
        letters = [c for c in block if c.isalpha()]
        if len(block.split()) < 4 or not letters:
            continue
        if sum(c.isupper() for c in letters) > 0.5 * len(letters):
            continue  # headings, tables of contents
        result.append(block)
    return result


def utterances(book: int, rng: random.Random) -> list[str]:
    result = []
    for paragraph in paragraphs(book):
        sentences = []
        for piece in SENTENCE_END.split(paragraph):
            piece = piece.strip()
            if sentences and ABBREVIATION.search(sentences[-1]):
                sentences[-1] += " " + piece  # "Mr. Bingley" is one sentence
            elif piece:
                sentences.append(piece)
        index = 0
        while index < len(sentences):
            # Mostly benchmark-like lengths, with short and very long tails.
            target = rng.choice((rng.randint(10, 80), rng.randint(40, 330), rng.randint(200, 460)))
            group = sentences[index]
            index += 1
            while index < len(sentences) and len(group) + 1 + len(sentences[index]) <= target:
                group += " " + sentences[index]
                index += 1
            if 8 <= len(group) <= 470:
                result.append(group)
    return result


_TOKENIZER = None


def phonemize_batch(texts: list[str]) -> list[tuple[str, list[int]]]:
    global _TOKENIZER
    if _TOKENIZER is None:
        from kokoro_onnx.tokenizer import Tokenizer

        _TOKENIZER = Tokenizer()
    # Same call as kokoro_onnx Tokenizer.phonemize, one text at a time.
    result = []
    for text in texts:
        phonemes = _TOKENIZER.phonemize(text, "en-us")
        result.append((phonemes, _TOKENIZER.tokenize(phonemes, limit=None)))
    return result


def phonemize_all(texts: list[str]) -> list[tuple[str, list[int]]]:
    batches = [texts[i : i + 200] for i in range(0, len(texts), 200)]
    with ProcessPoolExecutor(max_workers=10) as pool:
        return [item for batch in pool.map(phonemize_batch, batches) for item in batch]


def check_frozen() -> None:
    """The training pipeline must reproduce the frozen chunks' token ids."""
    items = json.loads(MANIFEST.read_text())["items"]
    for item in items:
        if len(item["chunks"]) != 1:
            continue  # the 30 s text was split by kokoro-onnx's chunker
        chunk = item["chunks"][0]
        phonemes, ids = phonemize_batch([item["text"]])[0]
        expected = chunk["input_ids"][1:-1]
        print(f"frozen {item['id']}: ids_match={ids == expected} tokens={len(ids)}")
        if ids != expected:
            raise RuntimeError(f"phonemizer differs on {item['id']}")


def build(name: str, books: tuple[int, ...], per_book: int, seed: int) -> None:
    rng = random.Random(seed)
    frozen_texts = [item["text"] for item in json.loads(MANIFEST.read_text())["items"]]
    rows = []
    for book in books:
        texts = utterances(book, rng)
        rng.shuffle(texts)
        texts = texts[:per_book]
        for text, (phonemes, ids) in zip(texts, phonemize_all(texts)):
            if not 3 <= len(ids) <= 508:
                continue
            if any(text in frozen or frozen[:40] in text for frozen in frozen_texts):
                continue
            rows.append({"book": book, "text": text, "phonemes": phonemes, "ids": ids})
        print(f"{name} book {book}: {sum(r['book'] == book for r in rows)} utterances")
    rng.shuffle(rows)
    path = CORPUS / f"{name}.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    lengths = sorted(len(r["ids"]) for r in rows)
    print(
        f"{name}: {len(rows)} utterances, {sum(lengths)} tokens, "
        f"median {lengths[len(lengths) // 2]}, max {lengths[-1]}"
    )


def main() -> None:
    check_frozen()
    build("holdout", HOLDOUT_BOOKS, PER_HOLDOUT_BOOK, seed=1)
    build("train", TRAIN_BOOKS, PER_TRAIN_BOOK, seed=0)


if __name__ == "__main__":
    main()
