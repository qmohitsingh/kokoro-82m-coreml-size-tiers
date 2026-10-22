"""Fine-tune the w256-b student on the recipe corpus.

Based on s014_train.py stage A (feature losses only, no teacher at train time):
    cache (teacher text_features, duration_hidden, durations; per voice)
    -> student (same architecture as w256-b, init from its step10000.pt)
    -> per-token weighted losses, normalized by the sum of token weights
       (every token counts once; sounds are not averaged over frames)
       total = w_text*text + w_hidden*hidden + w_dur*duration
    token weight = 1, or --w-target on the --targets tokens (SYM or SYM=weight)
-> <work>/runs/<name>/{args.json, log.jsonl, last.pt, step<N>.pt}

    python scripts/s018_train.py --work runs-s018 --voice-dir voices --init student-w256-b-step10000.pt --name run1 --steps 3000
    # S018 student recipe: add  --targets ʧ ʤ A I O W Y --stage b --down-count 4 --lr 2e-4
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s014_teacher as teacher  # noqa: E402
import s014_train as base  # noqa: E402
import s018_text as T  # noqa: E402
from s014_student import Student, parameter_count  # noqa: E402
from s018_cache_teacher import DEFAULT_VOICES, load_voices  # noqa: E402

TARGETS = ("ʧ", "ʤ", "A", "I", "O", "W")


class Split(base.Split):
    """Same memory-mapped cache; each row has its own voice."""

    def __init__(self, root: Path):
        index = json.loads((root / "index.json").read_text())
        self.offsets = np.array(index["offsets"])
        self.lengths = np.diff(self.offsets)
        self.voices = index["voices"]
        load = lambda n: np.load(root / f"{n}.npy", mmap_mode="r")
        self.ids, self.duration = load("ids"), load("duration")
        self.hidden_q, self.hidden_scale = load("hidden_q"), load("hidden_scale")
        self.text_q, self.text_scale = load("text_q"), load("text_scale")

    def batch(self, indices, voices: dict, device: str) -> dict:
        batch = super().batch(indices, next(iter(voices.values())), device)  # style replaced below
        style = np.stack([voices[self.voices[i]][int(self.lengths[i]) - 3] for i in indices])
        batch["style"] = torch.from_numpy(style.astype(np.float32)).to(device)
        return batch


def parse_targets(specs, w_target: float) -> dict[int, float]:
    """['ʧ', 'A=6'] -> {vocab id: per-token weight}."""
    out = {}
    for s in specs:
        sym, _, w = s.partition("=")
        out[T.VOCAB[sym]] = float(w) if w else w_target
    return out


def token_weights(ids: torch.Tensor, weights: dict[int, float]) -> torch.Tensor:
    w = torch.ones_like(ids, dtype=torch.float32)
    for i, x in weights.items():
        w = torch.where(ids == i, torch.full_like(w, x), w)
    return w


def wmean(x, w):
    return (x * w).sum() / w.sum().clamp(min=1e-6)


def losses(out, batch, norms, weights, w_total):
    hidden_s, logits, text_s = out
    w = token_weights(batch["ids"], weights) * batch["keep"]
    d_s, d_t = torch.sigmoid(logits).sum(-1), batch["duration"]
    rounded = torch.round(d_t).clamp(min=1)
    edge = F.relu((d_s - rounded).abs() - 0.35)
    duration = wmean((d_s - d_t).abs() + 2.0 * edge, w)
    if w_total:
        ts, tt = (d_s * batch["keep"]).sum(-1), (d_t * batch["keep"]).sum(-1)
        duration = duration + w_total * ((ts - tt).abs() / tt).mean()
    hidden = wmean((((hidden_s[..., :512] - batch["hidden"]) / norms["hidden"]) ** 2).mean(-1), w)
    text = wmean((((text_s.transpose(1, 2) - batch["text"]) / norms["text"]) ** 2).mean(-1), w)
    return dict(duration=duration, hidden=hidden, text=text)


@torch.no_grad()
def evaluate(student, split, voices, device, weights):
    """Validation on the held-out cache: overall and on target-token rows."""
    student.eval()
    acc = {k: np.zeros(2) for k in ("exact", "text_err", "text_sig", "hid_err", "hid_sig", "n")}
    ids_t = torch.tensor(list(weights), device=device)
    for start in range(0, len(split), 64):
        idx = list(range(start, min(start + 64, len(split))))
        b = split.batch(idx, voices, device)
        hs, lg, ts = student(b["ids"], b["style"], b["keep"][..., None])
        keep = b["keep"]
        tgt = torch.isin(b["ids"], ids_t).float() * keep
        for k, m in enumerate((keep, tgt)):
            exact = (torch.round(torch.sigmoid(lg).sum(-1)).clamp(min=1) == torch.round(b["duration"]).clamp(min=1)).float()
            acc["exact"][k] += float((exact * m).sum())
            acc["n"][k] += float(m.sum())
            acc["text_err"][k] += float((((ts.transpose(1, 2) - b["text"]) ** 2).sum(-1) * m).sum())
            acc["text_sig"][k] += float(((b["text"] ** 2).sum(-1) * m).sum())
            acc["hid_err"][k] += float((((hs[..., :512] - b["hidden"]) ** 2).sum(-1) * m).sum())
            acc["hid_sig"][k] += float(((b["hidden"] ** 2).sum(-1) * m).sum())
    student.train()
    snr = lambda s, e: 10 * math.log10(s / e)
    r = {}
    for k, name in enumerate(("all", "target")):
        n = max(acc["n"][k], 1)
        r[name] = dict(
            tokens=int(acc["n"][k]), duration_exact=acc["exact"][k] / n,
            text_snr=snr(max(acc["text_sig"][k], 1e-9), max(acc["text_err"][k], 1e-9)),
            hidden_snr=snr(max(acc["hid_sig"][k], 1e-9), max(acc["hid_err"][k], 1e-9)))
    return r


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True, help="work dir with cache/<split> and heldout; runs go to <work>/runs/<name>")
    p.add_argument("--voice-dir", type=Path, required=True, help="folder with <voice>_f32.bin tables")
    p.add_argument("--voices", nargs="+", default=list(DEFAULT_VOICES))
    p.add_argument("--init", type=Path, required=True, help="starting checkpoint, e.g. the published student-w256-b-step10000.pt")
    p.add_argument("--vocab", type=Path, help="teacher config.json (default models/pytorch/config.json)")
    p.add_argument("--name", required=True)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-tokens", type=int, default=8000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--w-text", type=float, default=3.0)
    p.add_argument("--w-hidden", type=float, default=2.0)
    p.add_argument("--w-dur", type=float, default=1.0)
    p.add_argument("--targets", nargs="+", default=list(TARGETS), help="weak tokens: SYM, or SYM=weight to override --w-target")
    p.add_argument("--w-target", type=float, default=4.0, help="per-token weight on the --targets tokens")
    p.add_argument("--w-total", type=float, default=10.0, help="utterance-total duration term (as w256-b)")
    p.add_argument("--stage", choices=("a", "b"), default="a", help="b adds pitch/energy and log-mel terms (frozen teacher pitch, decoder, generator)")
    p.add_argument("--down-count", type=int, default=8)
    p.add_argument("--crop", type=int, default=80)
    p.add_argument("--w-pitch", type=float, default=1.0)
    p.add_argument("--w-mel", type=float, default=1.0)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-split", default="train", help="train cache dir name under <work>/cache")
    args = p.parse_args()
    T.set_vocab(args.vocab)

    run = args.work / "runs" / args.name
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(args), default=str, indent=2))
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    train, val = Split(args.work / "cache" / args.train_split), Split(args.work / "cache/heldout")
    weights = parse_targets(args.targets, args.w_target)
    voices = load_voices(args.voice_dir, args.voices)
    norms = {k: torch.from_numpy(v).to(args.device) for k, v in base.feature_norms(train).items()}

    saved = torch.load(args.init, map_location="cpu")
    student = Student(**saved["config"])
    student.load_state_dict(saved["model"])
    student = student.to(args.device).train()
    print(f"student parameters {parameter_count(student)} config {saved['config']}", flush=True)

    model = teacher.load_teacher(args.device) if args.stage == "b" else None
    logmel = base.LogMel(args.device) if args.stage == "b" else None
    def val_metrics():
        ev = evaluate(student, val, voices, args.device, weights)
        if args.stage == "b":
            ev.update(base.evaluate_downstream(model, student, val, voices, args.device, logmel, args.crop))
        return ev
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    log = open(run / "log.jsonl", "a")
    def emit(row):
        print(json.dumps(row), flush=True); log.write(json.dumps(row) + "\n"); log.flush()
    emit({"step": 0, "eval": val_metrics()})
    step, start = 0, time.time()
    while step < args.steps:
        for indices in base.buckets(train, args.batch_tokens, rng):
            if step >= args.steps:
                break
            b = train.batch(indices, voices, args.device)
            out = student(b["ids"], b["style"], b["keep"][..., None])
            ls = losses(out, b, norms, weights, args.w_total)
            total = args.w_dur * ls["duration"] + args.w_hidden * ls["hidden"] + args.w_text * ls["text"]
            if args.stage == "b":
                down = base.downstream_losses(model, out, b, logmel, args.down_count, args.crop, rng)
                if down is not None:
                    ls.update(down)
                    total = total + args.w_pitch * (down["f0"] + down["energy"]) + args.w_mel * down["logmel"]
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 50 == 0:
                emit({"step": step, "sec": round(time.time() - start), "lr": sched.get_last_lr()[0], **{k: round(float(v), 5) for k, v in ls.items()}})
            if step % args.eval_every == 0 or step == args.steps:
                ev = val_metrics()
                emit({"step": step, "eval": ev})
                state = {"config": student.config, "model": student.state_dict(), "step": step, "eval": ev}
                torch.save(state, run / "last.pt"); torch.save(state, run / f"step{step}.pt")
    print(f"done {step} steps, {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
