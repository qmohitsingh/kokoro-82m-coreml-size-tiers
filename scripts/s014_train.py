"""Distil Kokoro's text body and duration path into the S014 student.

    cache (teacher targets) -> length-bucketed batches of ~N tokens
    stage A: duration + text_feature + prosody losses
    stage B: + pitch_energy (frozen F0Ntrain) + multi-resolution log-mel
             (frozen decoder and generator on 2 s crops, shared noise seed)
    every --eval-every steps: validation set only (never the hidden test)
    -> artifacts/s014/runs/<name>/{last,best}.pt, log.jsonl

    .venv-convert/bin/python scripts/s014_train.py \
        --name w256-a --stage a --steps 40000
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
from s014_student import Student, parameter_count  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts/s014/cache"


class Split:
    """Memory-mapped teacher targets for one corpus split."""

    def __init__(self, name: str):
        root = CACHE / name
        index = json.loads((root / "index.json").read_text())
        self.offsets = np.array(index["offsets"])
        self.lengths = np.diff(self.offsets)
        load = lambda n: np.load(root / f"{n}.npy", mmap_mode="r")
        self.ids, self.duration = load("ids"), load("duration")
        self.hidden_q, self.hidden_scale = load("hidden_q"), load("hidden_scale")
        self.text_q, self.text_scale = load("text_q"), load("text_scale")

    def __len__(self) -> int:
        return len(self.lengths)

    def batch(self, indices, voice: np.ndarray, device: str) -> dict:
        L = int(self.lengths[indices].max())
        B = len(indices)
        ids = np.zeros((B, L), np.int64)
        duration = np.zeros((B, L), np.float32)
        hidden = np.zeros((B, L, 512), np.float32)
        text = np.zeros((B, L, 512), np.float32)
        for j, i in enumerate(indices):
            o, n = int(self.offsets[i]), int(self.lengths[i])
            ids[j, :n] = self.ids[o : o + n]
            duration[j, :n] = self.duration[o : o + n]
            hidden[j, :n] = self.hidden_q[o : o + n] * self.hidden_scale[o : o + n, None].astype(np.float32)
            text[j, :n] = self.text_q[o : o + n] * self.text_scale[o : o + n, None].astype(np.float32)
        lengths = self.lengths[indices]
        style = teacher.style_rows(voice, [int(n) - 2 for n in lengths])
        keep = (np.arange(L)[None] < lengths[:, None]).astype(np.float32)
        to = lambda a: torch.from_numpy(a).to(device)
        return dict(
            ids=to(ids), duration=to(duration), hidden=to(hidden), text=to(text),
            style=to(style.astype(np.float32)), keep=to(keep), lengths=to(lengths),
        )


def buckets(split: Split, batch_tokens: int, rng: random.Random) -> list[list[int]]:
    """Shuffle, sort within chunks of 100 by length, cut to a token budget."""
    order = list(range(len(split)))
    rng.shuffle(order)
    batches = []
    for start in range(0, len(order), 100 * 32):
        chunk = sorted(order[start : start + 100 * 32], key=lambda i: split.lengths[i])
        current = []
        for i in chunk:
            if current and (len(current) + 1) * split.lengths[i] > batch_tokens:
                batches.append(current)
                current = []
            current.append(i)
        batches.append(current)
    rng.shuffle(batches)
    return batches


def masked_mean(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    return (x * keep).sum() / keep.sum().clamp(min=1)


def feature_losses(out, batch, norms, w_total: float = 0.0) -> dict:
    hidden_s, logits, text_s = out
    keep = batch["keep"]
    d_s = torch.sigmoid(logits).sum(-1)
    d_t = batch["duration"]
    rounded = torch.round(d_t).clamp(min=1)
    # L1 on the continuous value, plus a push to stay inside the teacher's
    # rounding interval (|d - round(d_t)| < 0.5), measured with a 0.15 margin.
    edge = F.relu((d_s - rounded).abs() - 0.35)
    duration = masked_mean((d_s - d_t).abs() + 2.0 * edge, keep)
    if w_total:
        # Per-token errors that lean one way add up over an utterance.
        total_s, total_t = (d_s * keep).sum(-1), (d_t * keep).sum(-1)
        duration = duration + w_total * ((total_s - total_t).abs() / total_t).mean()
    hidden = masked_mean((((hidden_s[..., :512] - batch["hidden"]) / norms["hidden"]) ** 2).mean(-1), keep)
    text = masked_mean((((text_s.transpose(1, 2) - batch["text"]) / norms["text"]) ** 2).mean(-1), keep)
    return dict(duration=duration, hidden=hidden, text=text)


class LogMel(torch.nn.Module):
    """Multi-resolution log-mel (same mel formula as scripts/logmel_dtw.py)."""

    def __init__(self, device: str, sizes=((512, 128), (1024, 256), (2048, 512)), mels=80):
        super().__init__()
        self.sizes = sizes
        self.filters = []
        for n_fft, _ in sizes:
            to_mel = lambda hz: 2595 * math.log10(1 + hz / 700)
            edges = 700 * (10 ** (torch.linspace(0, to_mel(12000), mels + 2) / 2595) - 1)
            bins = torch.linspace(0, 12000, n_fft // 2 + 1)
            low, mid, high = edges[:-2, None], edges[1:-1, None], edges[2:, None]
            fb = torch.clamp(torch.minimum((bins - low) / (mid - low), (high - bins) / (high - mid)), min=0)
            self.filters.append(fb.to(device))
        self.windows = [torch.hann_window(n, device=device) for n, _ in sizes]

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        loss = 0
        for (n_fft, hop), fb, window in zip(self.sizes, self.filters, self.windows):
            spec = lambda x: torch.stft(x, n_fft, hop, window=window, return_complex=True).abs() ** 2
            ma = torch.log(fb @ spec(a) + 1e-5)
            mb = torch.log(fb @ spec(b) + 1e-5)
            loss = loss + (ma - mb).abs().mean()
        return loss / len(self.sizes)


def downstream_losses(model, out, batch, logmel, count: int, crop: int, rng: random.Random) -> dict | None:
    """Pitch/energy and log-mel on `count` utterances, aligned with the teacher's
    rounded durations. Each utterance contributes one `crop`-frame window; the
    teacher and the student both run pitch and decoder on that same window, as
    one batch, with the same noise seed."""
    hidden_s, _, text_s = out
    windows = []
    order = list(range(hidden_s.shape[0]))
    rng.shuffle(order)
    for j in order:
        n = int(batch["lengths"][j])
        frames = torch.round(batch["duration"][j, :n]).clamp(min=1).long()
        total = int(frames.sum())
        if total < crop:
            continue
        aln = teacher.alignment(frames, n, total)
        a = rng.randint(0, total - crop)
        aln = aln[:, a : a + crop]
        windows.append((
            batch["hidden"][j, :n].T @ aln, hidden_s[j, :n, :512].T @ aln,
            batch["text"][j, :n].T @ aln, text_s[j, :, :n] @ aln, j,
        ))
        if len(windows) == count:
            break
    if not windows:
        return None
    h_t, h_s, asr_t, asr_s, rows = zip(*windows)
    style = batch["style"][list(rows)]
    s = style[:, 128:]
    copy = s[:, :, None].expand(-1, -1, crop)
    h_t = torch.cat([torch.stack(h_t), copy], 1)
    h_s = torch.cat([torch.stack(h_s), copy], 1)
    with torch.no_grad():
        f0_t, n_t = model.predictor.F0Ntrain(h_t, s)
    f0_s, n_s = model.predictor.F0Ntrain(h_s, s)
    voiced = (f0_t > 10).float()
    f0_loss = ((f0_s - f0_t).abs() * voiced).sum() / voiced.sum().clamp(min=1) / 100
    n_loss = (n_s - n_t).abs().mean()

    seed = rng.randint(0, 2**31 - 1)
    torch.manual_seed(seed)
    with torch.no_grad():
        audio_t = model.decoder(torch.stack(asr_t), f0_t, n_t, style[:, :128])
    torch.manual_seed(seed)
    audio_s = model.decoder(torch.stack(asr_s), f0_s, n_s, style[:, :128])
    mel_loss = logmel(audio_s.reshape(len(rows), -1), audio_t.reshape(len(rows), -1))
    return dict(f0=f0_loss, energy=n_loss, logmel=mel_loss)


@torch.no_grad()
def evaluate(student, split: Split, voice, device, norms, limit: int | None = None) -> dict:
    student.eval()
    order = sorted(range(len(split)), key=lambda i: split.lengths[i])[: limit or len(split)]
    stats = dict(tokens=0, exact=0, abs_frames=0.0, hidden_err=0.0, hidden_sig=0.0,
                 text_err=0.0, text_sig=0.0, total_rel=[], total_abs=[])
    for start in range(0, len(order), 64):
        idx = order[start : start + 64]
        batch = split.batch(idx, voice, device)
        hidden_s, logits, text_s = student(batch["ids"], batch["style"], batch["keep"][..., None])
        keep = batch["keep"].bool()
        f_s = torch.round(torch.sigmoid(logits).sum(-1)).clamp(min=1)
        f_t = torch.round(batch["duration"]).clamp(min=1)
        stats["tokens"] += int(keep.sum())
        stats["exact"] += int(((f_s == f_t) & keep).sum())
        stats["abs_frames"] += float(((f_s - f_t).abs() * keep).sum())
        for j in range(len(idx)):
            ts, tt = float((f_s[j] * keep[j]).sum()), float((f_t[j] * keep[j]).sum())
            stats["total_rel"].append(abs(ts - tt) / tt)
            stats["total_abs"].append(abs(ts - tt))
        m = keep[..., None].float()
        stats["hidden_err"] += float((((hidden_s[..., :512] - batch["hidden"]) * m) ** 2).sum())
        stats["hidden_sig"] += float(((batch["hidden"] * m) ** 2).sum())
        stats["text_err"] += float((((text_s.transpose(1, 2) - batch["text"]) * m) ** 2).sum())
        stats["text_sig"] += float(((batch["text"] * m) ** 2).sum())
    student.train()
    return dict(
        duration_exact=stats["exact"] / stats["tokens"],
        frames_mae=stats["abs_frames"] / stats["tokens"],
        total_rel_median=float(np.median(stats["total_rel"])),
        total_abs_mean=float(np.mean(stats["total_abs"])),
        hidden_snr=10 * math.log10(stats["hidden_sig"] / stats["hidden_err"]),
        text_snr=10 * math.log10(stats["text_sig"] / stats["text_err"]),
    )


@torch.no_grad()
def evaluate_downstream(model, student, split: Split, voice, device, logmel, crop: int) -> dict:
    """Stage-B losses on fixed validation windows (same windows and seeds every time)."""
    student.eval()
    rng = random.Random(123)
    sums = {}
    batches = buckets(split, 8000, rng)[:24]
    for indices in batches:
        batch = split.batch(indices, voice, device)
        out = student(batch["ids"], batch["style"], batch["keep"][..., None])
        down = downstream_losses(model, out, batch, logmel, 8, crop, rng)
        for k, v in (down or {}).items():
            sums[k] = sums.get(k, 0.0) + float(v) / len(batches)
    student.train()
    return {f"val_{k}": v for k, v in sums.items()}


def feature_norms(split: Split) -> dict:
    """Per-channel std of the teacher features, from 2,000 utterances."""
    rng = np.random.default_rng(0)
    rows = rng.choice(len(split), min(2000, len(split)), replace=False)
    pick = np.concatenate([np.arange(split.offsets[i], split.offsets[i + 1]) for i in rows])
    pick = np.sort(pick)
    h = split.hidden_q[pick] * split.hidden_scale[pick, None].astype(np.float32)
    t = split.text_q[pick] * split.text_scale[pick, None].astype(np.float32)
    return dict(hidden=h.std(0) + 1e-3, text=t.std(0) + 1e-3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--stage", choices=("a", "b"), default="a")
    parser.add_argument("--init", type=Path, help="checkpoint to start from (stage b)")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layout", default="ccaccacc")
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--batch-tokens", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--down-count", type=int, default=8, help="utterances per step for stage-b losses")
    parser.add_argument("--crop", type=int, default=80, help="pitch and decoder window, frames (80 = 2 s)")
    parser.add_argument("--w-pitch", type=float, default=1.0)
    parser.add_argument("--w-mel", type=float, default=1.0)
    parser.add_argument("--w-total", type=float, default=0.0, help="utterance-total duration, relative L1")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--train-split", default="train", help="holdout only for code smoke tests")
    args = parser.parse_args()

    run = ROOT / "artifacts/s014/runs" / args.name
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(args), default=str, indent=2))
    torch.manual_seed(0)
    rng = random.Random(0)

    train, val = Split(args.train_split), Split("holdout")
    voice = teacher.load_voice()
    norms = {k: torch.from_numpy(v).to(args.device) for k, v in feature_norms(train).items()}

    if args.init:
        saved = torch.load(args.init, map_location="cpu")
        student = Student(**saved["config"])
        student.load_state_dict(saved["model"])
    else:
        student = Student(width=args.width, layout=args.layout)
    student = student.to(args.device).train()
    print(f"student parameters {parameter_count(student)}", flush=True)

    model = teacher.load_teacher(args.device) if args.stage == "b" else None
    logmel = LogMel(args.device) if args.stage == "b" else None

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    schedule = lambda step: min(1, (step + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(step, args.steps) / args.steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    best = -1e9
    step, epoch = 0, 0
    start = time.time()
    log = open(run / "log.jsonl", "a")
    while step < args.steps:
        for indices in buckets(train, args.batch_tokens, rng):
            if step >= args.steps:
                break
            batch = train.batch(indices, voice, args.device)
            out = student(batch["ids"], batch["style"], batch["keep"][..., None])
            losses = feature_losses(out, batch, norms, args.w_total)
            total = losses["duration"] + losses["hidden"] + losses["text"]
            if args.stage == "b":
                down = downstream_losses(model, out, batch, logmel, args.down_count, args.crop, rng)
                if down is not None:
                    losses.update(down)
                    total = total + args.w_pitch * (down["f0"] + down["energy"]) + args.w_mel * down["logmel"]
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 100 == 0:
                row = {"step": step, "epoch": epoch, "sec": round(time.time() - start),
                       "lr": scheduler.get_last_lr()[0], **{k: round(float(v), 5) for k, v in losses.items()}}
                print(json.dumps(row), flush=True)
                log.write(json.dumps(row) + "\n")
                log.flush()
            if step % args.eval_every == 0 or step == args.steps:
                metrics = evaluate(student, val, voice, args.device, norms)
                if args.stage == "b":
                    metrics.update(evaluate_downstream(model, student, val, voice, args.device, logmel, args.crop))
                # Exact rounded durations matter most; features break ties.
                score = metrics["duration_exact"] * 100 + 0.5 * (metrics["hidden_snr"] + metrics["text_snr"])
                row = {"step": step, "eval": metrics, "score": round(score, 4)}
                print(json.dumps(row), flush=True)
                log.write(json.dumps(row) + "\n")
                log.flush()
                saved = {"config": student.config, "model": student.state_dict(), "step": step, "eval": metrics}
                torch.save(saved, run / "last.pt")
                if args.stage == "b":
                    # Stage B is chosen later by audio diagnostics, so keep every evaluation.
                    torch.save(saved, run / f"step{step}.pt")
                if score > best:
                    best = score
                    torch.save(saved, run / "best.pt")
        epoch += 1
    print(f"done {step} steps, {time.time() - start:.0f}s, best score {best:.4f}")


if __name__ == "__main__":
    main()
