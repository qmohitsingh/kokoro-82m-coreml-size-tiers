"""Held-out phoneme set for the retraining recipe. Build and freeze it BEFORE the training corpus.

Candidates: single sentences (20-130 chars, <= 22 words) from your own text files
that are never used for training, plus a few hand-written sentences for punctuation
tokens. Phonemized with the misaki frontend (s018_text.py). Greedy pick until every
token that English text can produce is covered at least --min-count times (or as
often as the candidates allow), then fill to --min-sentences.

Writes <work>/heldout/{sentences.txt, tokens.json, counts.json, SHA256}. Tokens in
Kokoro's vocab that the English frontend never emits (non-English phonemes) are
listed in counts.json under "unreachable".

    python scripts/s018_heldout.py --work runs-s018 --text heldout_book1.txt heldout_book2.txt
"""

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s018_text as T  # noqa: E402

HAND = [
    "The old bridge (built in 1902) is closed for repairs.",
    "She waited, and waited… and then the phone rang.",
    "He opened the box (carefully) and looked inside.",
    "Nobody knew where the keys were (not even Tom).",
    "The train was late again… so we walked home.",
    "We bought apples (red ones), bread, and a jar of honey.",
    "I thought about it for a while… maybe tomorrow.",
    "The garden (which my mother planted) is full of roses.",
    "“Come in,” she said, “the kettle is on.”",
    "He asked, “Where are you going?” and she did not answer.",
    "Wait — did you hear that?",
    "There was only one rule: never open the door at night.",
    "Pack warm clothes; the nights are cold, and the road is long.",
    "It was quiet… too quiet, really.",
    "Well… I suppose we could try again.",
    "The letter ended with a single word… goodbye.",
    "What a lovely morning! Shall we have breakfast outside?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True, help="work dir; output goes to <work>/heldout")
    ap.add_argument("--text", type=Path, nargs="+", required=True, help="plain-text files to draw held-out sentences from")
    ap.add_argument("--vocab", type=Path, help="teacher config.json (default models/pytorch/config.json)")
    ap.add_argument("--exclude", type=Path, nargs="*", default=[], help="files with test sentences (one per line) that must not be picked")
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--min-sentences", type=int, default=70)
    ap.add_argument("--seed", type=int, default=18)
    args = ap.parse_args()
    T.set_vocab(args.vocab)
    OUT = args.work / "heldout"
    OUT.mkdir(parents=True, exist_ok=True)
    MIN_COUNT, MIN_SENTENCES = args.min_count, args.min_sentences

    cands = sorted(set(T.read_sentences(args.text, 20, 130, 22)) | set(HAND))
    random.Random(args.seed).shuffle(cands)
    rows = []
    for s in cands:
        ph, ids = T.phonemize(s)
        if 8 <= len(ids) <= 120:
            rows.append((s, ph, ids, collections.Counter(ids)))
    total = collections.Counter()
    for r in rows:
        total.update(r[3])
    reachable = {i for i in total if total[i] > 0}
    unreachable = sorted(t for t, i in T.VOCAB.items() if i not in reachable)

    chosen, have = [], collections.Counter()
    # hand sentences first: they carry the punctuation tokens
    pool = list(range(len(rows)))
    for k in [i for i, r in enumerate(rows) if r[0] in HAND]:
        chosen.append(k); have.update(rows[k][3]); pool.remove(k)
    while True:
        # a token the candidates cannot supply MIN_COUNT times is covered as often as they allow
        deficit = {i: min(MIN_COUNT, total[i]) - have[i] for i in reachable if have[i] < min(MIN_COUNT, total[i])}
        if not deficit:
            break
        def gain(k):
            c = rows[k][3]
            return sum(min(c[i], d) for i, d in deficit.items() if i in c) / (len(rows[k][2]) ** 0.5)
        k = max(pool, key=gain)
        if gain(k) == 0:
            raise RuntimeError(f"cannot cover {[T.ID_TO_TOKEN[i] for i in deficit]}")
        chosen.append(k); have.update(rows[k][3]); pool.remove(k)
    rng = random.Random(5)
    while len(chosen) < MIN_SENTENCES and pool:
        k = rng.choice(pool); pool.remove(k); chosen.append(k); have.update(rows[k][3])

    banned = {l.strip() for p in args.exclude for l in open(p, encoding="utf-8") if l.strip()}
    items = []
    for k in sorted(chosen):
        s, ph, ids, _ = rows[k]
        assert s not in banned
        items.append(dict(text=s, phonemes=ph, ids=[0, *ids, 0], style=len(ids) - 1))
    (OUT / "sentences.txt").write_text("".join(i["text"] + "\n" for i in items))
    (OUT / "tokens.json").write_text(json.dumps(items, ensure_ascii=False, indent=1))
    counts = {T.ID_TO_TOKEN[i]: have[i] for i in sorted(have)}
    (OUT / "counts.json").write_text(json.dumps(dict(
        sentences=len(items), words=sum(len(i["text"].split()) for i in items),
        min_count=MIN_COUNT, min_over_reachable=min(have[i] for i in reachable),
        reachable_tokens=len(reachable), vocab_tokens=len(T.VOCAB),
        unreachable=unreachable, counts=counts), ensure_ascii=False, indent=1))
    digest = hashlib.sha256()
    for name in ("sentences.txt", "tokens.json", "counts.json"):
        digest.update((OUT / name).read_bytes())
    (OUT / "SHA256").write_text(
        "\n".join(f"{hashlib.sha256((OUT / n).read_bytes()).hexdigest()}  {n}" for n in ("sentences.txt", "tokens.json", "counts.json"))
        + f"\n{digest.hexdigest()}  combined\n")
    print(len(items), "sentences", sum(len(i["text"].split()) for i in items), "words")
    print("reachable", len(reachable), "of", len(T.VOCAB), "min count", min(have[i] for i in reachable))
    print("unreachable:", "".join(unreachable))


if __name__ == "__main__":
    main()
