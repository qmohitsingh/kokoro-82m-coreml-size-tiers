"""Templated training sentences for what old books rarely contain.

    numbers, dates, times, prices, percentages, phone numbers, short
    questions, names, and lists -> espeak-ng phonemes -> templates.jsonl

Training only. The hidden test sentences are written separately and never
pass through this script.

    .venv-convert/bin/python scripts/s014_templates.py
"""

import json
import random

from s014_corpus import CORPUS, phonemize_all

NAMES = (
    "Priya", "Mateo", "Aisha", "Kenji", "Olga", "Tomasz", "Ngozi", "Siobhan",
    "Raúl", "Leilani", "Dmitri", "Fatima", "Joaquin", "Ingrid", "Kwame",
    "Yuki", "Bartholomew", "Anneliese", "Oluwaseun", "Xavier", "Margarethe",
)
PLACES = (
    "Tucson", "Reykjavik", "Nairobi", "Brisbane", "Saskatoon", "Marseille",
    "Gdansk", "Cebu", "Valparaíso", "Albuquerque", "Edinburgh", "Kyoto",
)
THINGS = (
    "invoice", "package", "flight", "meeting", "prescription", "report",
    "concert", "lease", "order", "appointment", "shipment", "recipe",
)
MONTHS = (
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
)


def sentence(rng: random.Random) -> str:
    name, other = rng.sample(NAMES, 2)
    place = rng.choice(PLACES)
    thing = rng.choice(THINGS)
    templates = (
        lambda: f"{name} paid ${rng.randint(1, 999)}.{rng.randint(0, 99):02d} for the {thing}.",
        lambda: f"The {thing} arrives on {rng.choice(MONTHS)} {rng.randint(1, 28)}, {rng.randint(1890, 2049)}.",
        lambda: f"Call {name} at {rng.randint(200, 999)}-{rng.randint(200, 999)}-{rng.randint(1000, 9999)} after {rng.randint(1, 12)}:{rng.choice(('00', '15', '30', '45'))}.",
        lambda: f"Sales in {place} rose {rng.randint(1, 99)}.{rng.randint(0, 9)} percent last quarter.",
        lambda: f"Is the {thing} for {name}, or for {other}?",
        lambda: f"Did {name} really drive {rng.randint(2, 900)} miles to {place}?",
        lambda: f"Why would {name} cancel the {thing} now?",
        lambda: f"{name}, {other}, and {rng.choice(NAMES)} met in {place} at {rng.randint(1, 12)} o'clock.",
        lambda: f"Room {rng.randint(100, 999)} holds {rng.randint(2, 60)} people; the hall holds {rng.randint(100, 4000)}.",
        lambda: f"The {thing} weighs {rng.randint(1, 80)} kilograms, about {rng.randint(2, 170)} pounds.",
        lambda: f"\"Wait!\" {name} shouted. \"The {thing} is still in {place}!\"",
        lambda: f"Chapter {rng.randint(1, 40)}, verse {rng.randint(1, 50)}, was read aloud by {name}.",
        lambda: f"In {rng.randint(1700, 2030)}, {name} moved from {place} to {rng.choice(PLACES)}.",
        lambda: f"Is it {rng.randint(2, 99)} or {rng.randint(100, 9999)}? {name} wasn't sure.",
        lambda: f"Step {rng.randint(1, 9)}: mix {rng.randint(1, 4)} cups of flour with {rng.randint(1, 3)} eggs.",
    )
    return rng.choice(templates)()


def main() -> None:
    rng = random.Random(7)
    texts = []
    while len(texts) < 6_000:
        # One to three templated sentences per utterance.
        texts.append(" ".join(sentence(rng) for _ in range(rng.choice((1, 1, 2, 3)))))
    rows = [
        {"book": 0, "text": text, "phonemes": phonemes, "ids": ids}
        for text, (phonemes, ids) in zip(texts, phonemize_all(texts))
        if 3 <= len(ids) <= 508
    ]
    (CORPUS / "templates.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    print(f"templates: {len(rows)} utterances, {sum(len(r['ids']) for r in rows)} tokens")
    print(rows[0]["text"], "|", rows[0]["phonemes"])


if __name__ == "__main__":
    main()
