"""Distillation corpus for the retraining recipe, weighted toward the tokens the student gets wrong.

    targeted  ->  neutral everyday templates, your problem words in new sentences, optional
                  focus groups (word lists for one sound), and the sentences of your own text
                  that are densest in the target tokens
    general   ->  sentences sampled from your own text files, so other sounds do not regress

All text goes through the misaki frontend (s018_text.py). Every sentence in the
exclusion files (held-out set, test sets) is dropped by exact match, by normalized
text, and by normalized substring (either direction for sentences of 5+ words).
Run AFTER s018_heldout.py. Writes <work>/corpus/{train.jsonl, stats.json, dedupe.json};
train.jsonl rows: {source, text, phonemes, ids} with ids WITHOUT the two pads.

Config (JSON, see s018_corpus_config.example.json):
  target_tokens   {phoneme symbol: wanted occurrences}; targeted sentences are added until each is met
  problem_words   words to force into many new sentences (optional)
  names           first names for the templates (optional)
  focus_groups    [{name, count, nouns, verbs, adjectives}] extra sentences around one sound (optional)
  n_targeted, n_general, seed
  exclude         extra files of test sentences, one per line (optional)

    python scripts/s018_corpus.py --work runs-s018 --text book1.txt book2.txt --config s018_corpus_config.json
"""

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s018_text as T  # noqa: E402

NAMES = "Chuck Jack Jane Jill Jenny Charlie Chloe Kate George Julia Jasper Chase Nora Ben Lena Omar".split()

# word pools, grouped by what they share
N_CH = "chapter chimney chair cheese child children teacher kitchen chicken chest chart cherry champion chance channel charge".split()
N_J = "bridge judge jar job journey village page age cage stage edge jacket jewel jungle journal juice jug jam".split()
N_A = "day name way lake plate rain train plane wave cake game gate tray paper table maple neighbor".split()
N_I = "light night line kite bike sign time mile pile size price prize fire tire eyes pie sky".split()
N_O = "home road boat coat bone phone stone toast globe note rose goal soap pillow window yellow shadow".split()
N_W = "house cloud mouse crowd town tower flower couch mouth hound crown fountain mountain doubt".split()
NOUNS = N_CH + N_J + N_A + N_I + N_O + N_W + "book table letter story garden dinner river window music".split()
V_BASE = ("catch change choose watch reach teach search check judge join jump charge manage enjoy say stay play wait "
          "make take save pay shape try find like write ride smile decide arrive hide shine go hope show know own hold "
          "open close post shout count bow allow").split()
ADJ = "nice kind strong brave bright great whole proud loud good late slow cold old sweet round".split()
PLACES = "Chicago Oregon Georgia Jamaica Kenya Poland Japan Chile Dallas Boston Cheshire Jersey Norway".split()
TIMES = "today tonight tomorrow this morning on Monday after lunch before the game at noon last night".split()
SHORT = ["Okay.", "Wait.", "Wait here.", "Nice.", "Say it again.", "Thank you.", "Good day.", "Not now.", "No way.", "Hi there.", "Yes, I can.", "I like it.", "Go home.", "Open the door.", "Be quiet.", "My name is Jane.", "Hold the line.", "Take it slow.", "Come in.", "What time is it?", "How old are you?", "Why not?", "I know.", "Right now.", "Say what?", "Good night.", "Welcome home."]

PAST = {"catch": "caught", "say": "said", "wait": "waited", "make": "made", "take": "took", "find": "found", "go": "went", "know": "knew", "hold": "held", "choose": "chose", "teach": "taught", "write": "wrote", "ride": "rode", "show": "showed", "shine": "shone", "pay": "paid"}
def past(v): return PAST.get(v, v + ("d" if v.endswith("e") else "ed"))
def third(v): return v + ("es" if v.endswith(("ch", "sh", "s")) else "s") if not v.endswith("y") or v.endswith(("ay", "ow")) else v[:-1] + "ies"
def art(w): return ("an " if w[0] in "aeiou" else "a ") + w


def modern(rng, names, pools=None):
    nouns, verbs, adjs = pools or (NOUNS, V_BASE, ADJ)
    n, n2, nn = rng.choice(nouns), rng.choice(nouns), rng.choice(names)
    v, v2, a, p = rng.choice(verbs), rng.choice(V_BASE), rng.choice(adjs), rng.choice(PLACES)
    t = rng.choice(TIMES)
    F = (
        lambda: f"{nn} will {v} the {n} {t}.",
        lambda: f"Please {v} the {n} before you {v2}.",
        lambda: f"The {n} in {p} was very {a}.",
        lambda: f"Did you {v} {art(n)} when you were in {p}?",
        lambda: f"I can {v} the {n}, but I cannot {v2} the {n2}.",
        lambda: f"{nn} said the {n} was {a}, and I {past(v)} it {t}.",
        lambda: f"We {past(v)} the {n} and then we {past(v2)} the {n2}.",
        lambda: f"What do you think about the {a} {n}?",
        lambda: f"Why did {nn} {v} the {n} in {p}?",
        lambda: f"It is {a}, so {v} the {n} and {v2} the {n2}.",
        lambda: f"{nn} and {rng.choice(names)} will {v} {art(n)} {t}.",
        lambda: f"There is {art(a)} {n} by the {n2} in {p}.",
        lambda: f"Could you {v} the {n} for me?",
        lambda: f"My {n} is {a}, and the {n2} is too.",
        lambda: f"{nn} {third(v)} the {n} every day.",
        lambda: f"Let us {v} and {v2} the {a} {n}.",
        lambda: f"How {a} is the {n} in {p}?",
        lambda: f"Don't {v} the {n}; just {v2} the {n2}.",
        lambda: f"Is the {n} {a}, or is it the {n2}?",
        lambda: f"Only {nn} would {v} {art(n)} like that.",
    )
    return rng.choice(F)()


def focus(group, names, rng):
    """One sentence around a focus group's words (nouns, verbs, adjectives; missing lists fall back to the defaults)."""
    nouns, verbs, adjs = group.get("nouns") or NOUNS, group.get("verbs") or V_BASE, group.get("adjectives") or ADJ
    n, n2 = rng.choice(nouns), rng.choice(nouns + NOUNS)
    nn, v, v2, a = rng.choice(names), rng.choice(verbs), rng.choice(V_BASE), rng.choice(adjs + ADJ)
    p, t = rng.choice(PLACES), rng.choice(TIMES)
    F = (
        lambda: f"The {n} is {a} {t}.",
        lambda: f"{nn} will {v} the {n} in {p}.",
        lambda: f"What a {a} {n}, {nn}!",
        lambda: f"Did {nn} {v} the {n2}?",
        lambda: f"I saw {art(a)} {n} by the {n2}.",
        lambda: f"It is your choice, {nn}: the {n} or the {n2}.",
        lambda: f"There is {art(n)} on the {n2}, so {v} it.",
        lambda: f"{nn} made a point about the {n} {t}.",
        lambda: f"The {n} by the {n2} was not {a}.",
        lambda: f"Why did {nn} {v} the {n2} and {v2} the {n}?",
        lambda: f"We {v} the {n} and {rng.choice(V_BASE)} the {n2} {t}.",
        lambda: f"My friend {rng.choice(['loves', 'enjoys', 'needs', 'avoids'])} the {n}.",
        lambda: f"A {a} {n} stood in {p}.",
    )
    return rng.choice(F)()


def forced(word, names, rng):
    """One sentence with `word` used as-is (frames cover the start, middle and end of a sentence)."""
    nm, n, n2, a, v = rng.choice(names), rng.choice(NOUNS), rng.choice(NOUNS), rng.choice(ADJ), rng.choice(V_BASE)
    t, p = rng.choice(TIMES), rng.choice(PLACES)
    W = word[0].upper() + word[1:]
    F = (
        f"{W}, {nm}.", f"Did you say {word}, {nm}?", f"The {n} is {a}, so {word} it.", f"I will {v} the {n}, and {word} {t}.",
        f"{W} is a {a} word.", f"{nm} wrote the word {word} on the {n}.", f"What does {word} mean, {nm}?",
        f"Can you use {word} in a sentence about the {n}?", f"{W}! The {n} in {p} is {a}.", f"We talked about {word} and the {n2} {t}.",
        f"Now say {word} again, {nm}.", f"{nm} likes the {n}, and the {n2} too: {word}.",
    )
    return rng.choice(F)


def read_list(path) -> list[str]:
    """One sentence per line (.txt), or a JSON/JSONL list of strings or {"text": ...} rows."""
    p = Path(path)
    if p.suffix == ".json":
        raw = json.loads(p.read_text())
    elif p.suffix == ".jsonl":
        raw = [json.loads(l) for l in open(p) if l.strip()]
    else:
        return [l.strip() for l in open(p, encoding="utf-8") if l.strip()]
    return [r["text"] if isinstance(r, dict) else r for r in raw]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True, help="work dir; reads <work>/heldout, writes <work>/corpus")
    ap.add_argument("--text", type=Path, nargs="+", required=True, help="plain-text files: the general portion and the dense-sentence pool")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, help="teacher config.json (default models/pytorch/config.json)")
    ap.add_argument("--out-name", default="corpus", help="output dir name under <work>")
    args = ap.parse_args()
    T.set_vocab(args.vocab)
    cfg = json.loads(args.config.read_text())
    N_TARGETED, N_GENERAL = cfg.get("n_targeted", 5200), cfg.get("n_general", 7000)
    names = cfg.get("names") or NAMES
    problem = cfg.get("problem_words", [])
    groups = cfg.get("focus_groups", [])
    TARGETS = {s: T.VOCAB[s] for s in cfg.get("target_tokens", {})}
    WANT = cfg.get("target_tokens", {})
    OUT, HELDOUT = args.work / args.out_name, args.work / "heldout"
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.get("seed", 1820))

    for name in ("sentences.txt", "tokens.json", "counts.json", "SHA256"):
        assert (HELDOUT / name).exists(), "freeze the held-out set first (s018_heldout.py)"
    held = read_list(HELDOUT / "sentences.txt")
    other = [s for p in cfg.get("exclude", []) for s in read_list(p)]
    banned = {"heldout": held, "other": other}
    exact = {k: set(v) for k, v in banned.items()}
    norm = {k: {T.normalize(s) for s in v} for k, v in banned.items()}
    norm_all = sorted(set().union(*norm.values()))

    seen_norm, rows, dropped = set(), [], collections.Counter()
    def add(text, source):
        n = T.normalize(text)
        if text in exact["heldout"] or text in exact["other"]: dropped["exact"] += 1; return False
        if n in norm["heldout"] or n in norm["other"]: dropped["normalized"] += 1; return False
        if any(b in n for b in norm_all): dropped["contains_test_sentence"] += 1; return False
        if any(n in b for b in norm_all if len(n.split()) >= 5): dropped["inside_test_sentence"] += 1; return False
        if n in seen_norm: dropped["duplicate_in_corpus"] += 1; return False
        ph, ids = T.phonemize(text)
        if not 3 <= len(ids) <= 508: dropped["length"] += 1; return False
        seen_norm.add(n); rows.append(dict(source=source, text=text, phonemes=ph, ids=ids)); return True

    pool = T.read_sentences(args.text, 12, 320)
    rng.shuffle(pool)
    # sentences of the user's text that are densest in the target tokens, used inside the targeted portion
    dense = []
    if TARGETS:
        ids_of = set(TARGETS.values())
        scored = []
        for s in pool[:20000]:
            ids = T.phonemize(s)[1]
            if 3 <= len(ids) <= 508:
                scored.append((sum(i in ids_of for i in ids) / len(ids), s))
        dense = [s for d, s in sorted(scored, key=lambda x: -x[0]) if d > 0][: N_TARGETED // 3]
        rng.shuffle(dense)

    def targeted():
        r = rng.random()
        if r < 0.30 and problem:
            parts = [forced(rng.choice(problem), names, rng) for _ in range(rng.choice((1, 1, 1, 2, 2, 3)))]
        elif r < 0.70 or not dense:
            parts = [modern(rng, names) for _ in range(rng.choice((1, 1, 1, 2, 2, 3)))]
        elif r < 0.92:
            parts = [dense.pop()]
        else:
            parts = [rng.choice(SHORT) for _ in range(rng.choice((1, 1, 2)))]
        if problem and rng.random() < 0.3:
            parts.insert(rng.randrange(len(parts) + 1), forced(rng.choice(problem), names, rng))
        return " ".join(parts)

    got = 0
    while got < N_TARGETED:
        got += add(targeted(), "targeted")
    for g in groups:
        got = 0
        while got < g["count"]:
            parts = [focus(g, names, rng) for _ in range(rng.choice((1, 1, 1, 2, 2, 3)))]
            if problem and rng.random() < 0.25:
                parts.insert(rng.randrange(len(parts) + 1), forced(rng.choice(problem), names, rng))
            got += add(" ".join(parts), "targeted_" + g["name"])
    # top up: keep adding targeted sentences until every target token reaches its wanted count
    tok = collections.Counter(i for r in rows for i in r["ids"])
    extra = 0
    while WANT and any(tok[TARGETS[s]] < c for s, c in WANT.items()) and extra < 2 * N_TARGETED:
        before = len(rows)
        add(targeted(), "targeted")
        extra += 1
        if len(rows) > before:
            tok.update(rows[-1]["ids"])
    got = 0
    for text in pool:
        if got >= N_GENERAL: break
        got += add(text, "general")
    rng.shuffle(rows)
    (OUT / "train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    tok = collections.Counter(); words = collections.Counter()
    for r in rows:
        tok.update(r["ids"]); words.update(re.findall(r"[a-z']+", r["text"].lower()))
    reach = json.loads((HELDOUT / "counts.json").read_text())["counts"]
    stats = dict(
        utterances=len(rows), by_source=dict(collections.Counter(r["source"] for r in rows)),
        tokens=sum(len(r["ids"]) for r in rows),
        target_token_counts={s: tok[i] for s, i in TARGETS.items()},
        target_token_wanted=WANT,
        problem_word_counts={w: words[w.lower()] for w in problem},
        min_count_any_reachable_token={t: tok[T.VOCAB[t]] for t in reach if t in T.VOCAB and tok[T.VOCAB[t]] < 20},
        token_counts={T.ID_TO_TOKEN[i]: c for i, c in sorted(tok.items())})
    (OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    (OUT / "dedupe.json").write_text(json.dumps(dict(
        heldout_sentences=len(held), other_test_sentences=len(other), dropped_candidates=dict(dropped),
        final_exact_matches=sum(r["text"] in exact["heldout"] | exact["other"] for r in rows),
        final_normalized_matches=sum(T.normalize(r["text"]) in norm["heldout"] | norm["other"] for r in rows),
        final_substring_matches=sum(any(b in T.normalize(r["text"]) for b in norm_all) for r in rows),
        heldout_sha256_combined=(HELDOUT / "SHA256").read_text().split()[-2]), indent=1))
    print(json.dumps({k: stats[k] for k in ("utterances", "by_source", "tokens", "target_token_counts", "problem_word_counts", "min_count_any_reachable_token")}, ensure_ascii=False))
    print((OUT / "dedupe.json").read_text())


if __name__ == "__main__":
    main()
