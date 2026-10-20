"""S015 v2: continue the v1 downstream students with adversarial and feature-matching losses.

Same distillation as s015_train.py (targets from the fp32 Kokoro-82M teacher on the
students' own inputs), plus HiFi-GAN discriminators that compare student audio with
fp32 teacher audio on the same crop and sine noise:

    init from v1 (weights + AdamW moments), new cosine schedule to --steps
    teacher F0 / energy: artifacts/s015/cache/pitch (s015_cache_pitch.py); live teacher pass
                         when either cache variant is incomplete or the cached frame count differs
    vocoder crop:        fresh random crop and fresh sine noise every step (audio not cached)
                         -> teacher logits + audio (fp32, TF32 off), student logits + audio
    discriminators:      MPD (periods 2, 3, 5, 7, 11) + MSD (3 scales), LSGAN, from --disc-start
    student loss:        f0 + energy + mel + mrstft + 0.5 logmag + complex           (v1, unchanged)
                         + ramp * (adv_weight * adversarial + fm_weight * feature matching)
                         ramp 0 before --adv-start, linear to 1 over --adv-ramp steps
    every --eval-every:  v1 validation (8 fixed holdout groups x {teacher, S014} inputs, crop 96)
                         + full chain on 16 fixed holdout utterances: S014 student front end ->
                           S015 vs fp32 downstream, same seed -> DTW log-mel, RMS, HF bands
    -> artifacts/s015/runs/<name>/{stepN.pt, best.pt, last.pt, log.jsonl, args.json, env.json}
    stops on NaN, divergence, low disk, --patience stale evaluations, or --steps

The holdout utterances used by the final S014 mixed8 comparison (rng 7) and the hidden
listening set are never used here.

    python scripts/s015_train_gan.py --name v2-a100 --init artifacts/s015/runs/v1/last.pt --device cuda
    python scripts/s015_train_gan.py --name v3-w80 --init artifacts/s015/runs/v3-w80/init.pt --steps 30000 \
        --gate 26000:2.3 --device cuda   # widened vocoder from s015_widen.py: v2 disc, fresh student AdamW
    python scripts/s015_train_gan.py --name v5-pitch --init artifacts/s015/runs/v5-pitch/init.pt --steps 8000 \
        --freeze-vocoder --eval-every 1000 --target-dtw 1.86 --lr 1e-4 --device cuda
        # pitch only: vocoder frozen, f0 + energy + vocoder losses through the frozen vocoder on the student's own
        # F0/N (no discriminators); every evaluation also runs the 15-item S014 comparison (s015-full arm, CPU),
        # which then decides best.pt, staleness, and the --target-dtw stop
    python scripts/s015_train_gan.py --name v6-pitch-only --init artifacts/s015/runs/v5-pitch/init.pt --steps 20000 \
        --freeze-vocoder --pitch-loss-only --eval-every 1000 --target-dtw 1.86 --lr 1e-5 --max-minutes 30 --device cuda
        # as v5-pitch but f0 + energy against the cached teacher only (no losses through the vocoder)
"""

import argparse
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s014_teacher as teacher  # noqa: E402
from logmel_dtw import dtw_logmel_db  # noqa: E402
from s014_student import Student  # noqa: E402
from s014_train import Split  # noqa: E402
from s015_cache_pitch import PitchCache  # noqa: E402
from s015_downstream import PitchStudent, VocoderStudent, parameter_count  # noqa: E402
from s015_eval import band_db, level_db, s015_audio, teacher_audio  # noqa: E402
from s015_train import (S014_STUDENT, Batcher, MultiResSTFT, SafeLogMel, crop_windows, disk_free_gb,  # noqa: E402
                        evaluate, head_audio, pitch_loss, pitch_teacher, score, sine_spectrum, teacher_logits,
                        vocoder_losses)

ROOT = Path(__file__).resolve().parents[1]


class ExactFP32:
    """Teacher passes and evaluation in full fp32 on CUDA (TF32 off); students train with TF32."""

    def __enter__(self):
        self.saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False

    def __exit__(self, *exc):
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = self.saved


class PeriodDiscriminator(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        chans = [1, 32, 128, 512, 1024]
        self.convs = nn.ModuleList(
            [weight_norm(nn.Conv2d(a, b, (5, 1), (3, 1), padding=(2, 0))) for a, b in zip(chans, chans[1:])]
            + [weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0)))])
        self.post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        b, c, t = x.shape
        if t % self.period:
            x = F.pad(x, (0, self.period - t % self.period), "reflect")
        x = x.view(b, c, -1, self.period)
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return x.flatten(1), feats


class ScaleDiscriminator(nn.Module):
    def __init__(self, norm=weight_norm):
        super().__init__()
        spec = [(1, 128, 15, 1, 1), (128, 128, 41, 2, 4), (128, 256, 41, 2, 16), (256, 512, 41, 4, 16),
                (512, 1024, 41, 4, 16), (1024, 1024, 41, 1, 16), (1024, 1024, 5, 1, 1)]
        self.convs = nn.ModuleList([norm(nn.Conv1d(a, b, k, s, groups=g, padding=k // 2)) for a, b, k, s, g in spec])
        self.post = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return x.flatten(1), feats


class Discriminators(nn.Module):
    """HiFi-GAN MPD + MSD on 24 kHz audio [B, T]."""

    def __init__(self):
        super().__init__()
        self.mpd = nn.ModuleList([PeriodDiscriminator(p) for p in (2, 3, 5, 7, 11)])
        self.msd = nn.ModuleList([ScaleDiscriminator(spectral_norm), ScaleDiscriminator(), ScaleDiscriminator()])
        self.pool = nn.AvgPool1d(4, 2, padding=2)

    def forward(self, audio):
        x = audio[:, None]
        outs = [d(x) for d in self.mpd]
        for i, d in enumerate(self.msd):
            if i:
                x = self.pool(x)
            outs.append(d(x))
        return outs


def d_loss(real, fake):
    return sum(((1 - r) ** 2).mean() + (f**2).mean() for (r, _), (f, _) in zip(real, fake))


def g_losses(real, fake):
    adv = sum(((1 - f) ** 2).mean() for f, _ in fake)
    fm = sum((rf.detach() - ff).abs().mean() for (_, rfs), (_, ffs) in zip(real, fake) for rf, ff in zip(rfs, ffs))
    return adv, fm


DTW_UTTERANCES = 16


@torch.no_grad()
def full_chain_eval(model, s014, pitch, vocoder, voice, device) -> dict:
    """16 fixed holdout utterances, full length: S014 student front end and durations ->
    S015 pitch + vocoder vs fp32 Kokoro downstream, same sine seed. Excludes the final
    comparison's holdout picks (rng 7)."""
    pitch.eval(), vocoder.eval()
    split = Split("holdout")
    exclude = set(np.random.default_rng(7).choice(len(split), 10, replace=False).tolist())
    pick = [int(i) for i in np.random.default_rng(11).permutation(len(split)) if int(i) not in exclude][:DTW_UTTERANCES]
    rows = []
    with ExactFP32():
        for k, i in enumerate(pick):
            o, n = int(split.offsets[i]), int(split.lengths[i])
            ids = torch.tensor([split.ids[o : o + n].astype(int).tolist()], device=device)
            style = torch.from_numpy(teacher.style_rows(voice, [n - 2]).astype(np.float32)).to(device)
            hidden, logits, text = s014(ids, style)
            frames = teacher.frames_from_logits(logits)[0]
            _, _, ref = teacher_audio(model, hidden, text, frames, style, 1234 + k)
            _, _, out = s015_audio(model, pitch, vocoder, hidden, text, frames, style, 1234 + k)
            bands = [a - r for a, r in zip(band_db(out), band_db(ref))]
            rows.append(dict(dtw=dtw_logmel_db(ref, out)[0], rms=level_db(out) - level_db(ref),
                             b6_8=bands[0], b8_10=bands[1], b10_12=bands[2], finite=bool(np.isfinite(out).all())))
    pitch.train(), vocoder.train()
    out = {k: float(np.mean([r[k] for r in rows])) for k in ("dtw", "rms", "b6_8", "b8_10", "b10_12")}
    out["finite"] = all(r["finite"] for r in rows)
    return out


def environment() -> dict:
    pip = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.split()
    keep = ("torch", "numpy", "kokoro", "misaki", "transformers", "soundfile")
    return {"host": platform.node(), "platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "packages": [p for p in pip if p.lower().startswith(keep)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--init", type=Path, help="weights (+ AdamW moments, discriminators if saved), new schedule")
    parser.add_argument("--resume", type=Path, help="this script's last.pt: everything")
    parser.add_argument("--steps", type=int, default=20000, help="final step (absolute)")
    parser.add_argument("--stop-after", type=int, default=0, help="stop after this many steps in this process (probe)")
    parser.add_argument("--utterances", type=int, default=8)
    parser.add_argument("--crops", default="80,96,112,128")
    parser.add_argument("--lr", type=float, default=2e-4, help="peak generator lr after the restart")
    parser.add_argument("--lr-d", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--floor", type=float, default=0.05, help="final lr as a share of peak")
    parser.add_argument("--student-share", type=float, default=0.5)
    parser.add_argument("--disc-start", type=int, default=3000, help="discriminators train from this step")
    parser.add_argument("--adv-start", type=int, default=4000, help="adversarial + FM losses reach the students from here")
    parser.add_argument("--adv-ramp", type=int, default=4000, help="steps to ramp them from 0 to full weight")
    parser.add_argument("--adv-weight", type=float, default=0.1)
    parser.add_argument("--fm-weight", type=float, default=0.2)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="stale evaluations before stopping (any new 15-item best when --freeze-vocoder; otherwise 0.5%%)",
    )
    parser.add_argument("--min-free-gb", type=float, default=10.0)
    parser.add_argument("--gate", help="STEP:DTW, stop at the first evaluation from STEP whose chain DTW is above DTW")
    parser.add_argument("--freeze-vocoder", action="store_true", help="train the pitch student only (see usage)")
    parser.add_argument("--pitch-loss-only", action="store_true", help="with --freeze-vocoder: f0 + energy losses only")
    parser.add_argument("--max-minutes", type=float, help="evaluate the current weights and stop after this long")
    parser.add_argument("--target-dtw", type=float, help="with --freeze-vocoder: stop once the 15-item full arm is <= this")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    assert bool(args.init) != bool(args.resume), "use exactly one of --init / --resume"
    assert args.freeze_vocoder or not args.pitch_loss_only, "--pitch-loss-only needs --freeze-vocoder"

    run = ROOT / "artifacts/s015/runs" / args.name
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(args), default=str, indent=2))
    (run / "env.json").write_text(json.dumps(environment(), indent=2))
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = not args.no_tf32
    torch.manual_seed(0)
    rng = random.Random(0)
    crops = [int(c) for c in args.crops.split(",")]

    model = teacher.load_teacher(args.device)
    voice = teacher.load_voice()
    saved = torch.load(S014_STUDENT, map_location="cpu")
    s014 = Student(**saved["config"])
    s014.load_state_dict(saved["model"])
    s014 = s014.to(args.device).eval()
    for p in s014.parameters():
        p.requires_grad_(False)
    train_split, val_split = Split("train"), Split("holdout")
    train = Batcher(train_split, voice, args.device, s014)
    val = Batcher(val_split, voice, args.device, s014)
    usable = [i for i in range(len(train_split)) if train_split.lengths[i] >= 40]
    cache = PitchCache(args.device) if PitchCache.available() else None  # incomplete/no cache: live teacher every step

    state = torch.load(args.init or args.resume, map_location="cpu")
    pitch = PitchStudent(**state.get("pitch_config") or {}).to(args.device)
    vocoder = VocoderStudent(**state.get("vocoder_config", {})).to(args.device)
    disc = Discriminators().to(args.device)
    params = [*pitch.parameters(), *vocoder.parameters()]
    if args.freeze_vocoder:
        vocoder.requires_grad_(False)
        params = [*pitch.parameters()]
    print(f"pitch {parameter_count(pitch)} vocoder {parameter_count(vocoder)} disc {parameter_count(disc)}", flush=True)
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.8, 0.99), weight_decay=0.0)
    optimizer_d = torch.optim.AdamW(disc.parameters(), lr=args.lr_d, betas=(0.8, 0.99), weight_decay=0.0)
    logmel, mrstft = SafeLogMel(args.device), MultiResSTFT(args.device)

    if args.init:
        pitch.load_state_dict(state["pitch"])
        vocoder.load_state_dict(state["vocoder"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])  # AdamW moments; the lr is reset below
        if "disc" in state:
            disc.load_state_dict(state["disc"])
            optimizer_d.load_state_dict(state["optimizer_d"])
        print(f"init: student AdamW {'loaded' if 'optimizer' in state else 'fresh'}, "
              f"discriminators {'loaded' if 'disc' in state else 'new'}", flush=True)
        start_step, best, stale, history = state["step"], float("inf"), 0, []
    else:
        pitch.load_state_dict(state["pitch"])
        vocoder.load_state_dict(state["vocoder"])
        disc.load_state_dict(state["disc"])
        optimizer.load_state_dict(state["optimizer"])
        optimizer_d.load_state_dict(state["optimizer_d"])
        start_step = state["start_step"]
        best, stale, history = state["best"], state["stale"], state["history"]
    if args.init:
        for opt, peak in ((optimizer, args.lr), (optimizer_d, args.lr_d)):
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] = peak
    span = args.steps - start_step

    def schedule(local):  # local = steps since start_step
        warm = min(1.0, (local + 1) / args.warmup)
        return warm * (args.floor + (1 - args.floor) * 0.5 * (1 + math.cos(math.pi * min(local, span) / span)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(optimizer_d, schedule)
    step = start_step
    if args.resume:
        scheduler.load_state_dict(state["scheduler"])
        scheduler_d.load_state_dict(state["scheduler_d"])
        step = state["step"]
    rng.seed(step)
    print(f"start_step {start_step} step {step} lr {scheduler.get_last_lr()[0]:.3e}", flush=True)

    log = open(run / "log.jsonl", "a")
    write = lambda row: (print(json.dumps(row), flush=True), log.write(json.dumps(row) + "\n"), log.flush())

    def weights():
        return {"pitch": pitch.state_dict(), "vocoder": vocoder.state_dict(), "pitch_config": pitch.config,
                "vocoder_config": vocoder.config, "step": step}

    def save_last():
        torch.save({**weights(), "disc": disc.state_dict(), "optimizer": optimizer.state_dict(),
                    "optimizer_d": optimizer_d.state_dict(), "scheduler": scheduler.state_dict(),
                    "scheduler_d": scheduler_d.state_dict(), "start_step": start_step, "best": best,
                    "stale": stale, "history": history}, run / "last.pt")

    def run_eval():
        nonlocal best, stale
        with ExactFP32():
            metrics = evaluate(model, pitch, vocoder, val, logmel, mrstft, crop=96)
        chain = full_chain_eval(model, s014, pitch, vocoder, voice, args.device)
        sc = score(metrics)
        compare = None
        if args.freeze_vocoder:  # 15-item S014 comparison, full arm, on CPU like every S015 comparison
            torch.save(weights(), run / f"step{step}.pt")
            done = subprocess.run([sys.executable, str(ROOT / "scripts/s015_compare_s014.py"), f"{args.name}-step{step}",
                                   "--arm", "s015-full", "--checkpoint", str(run / f"step{step}.pt")],
                                  capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            means = [line for line in done.stdout.splitlines() if line.startswith("mean ")]
            if not means:
                return f"15-item comparison failed: {done.stderr[-500:]}"
            compare = json.loads(means[-1][5:])
            sc = compare["dtw"]
        improved = sc < (best if compare else best * 0.995)  # 15-item arm: any new best counts
        first = not history
        history.append(sc)
        if sc < best:
            best = sc
        stale = 0 if improved or first else stale + 1
        write({"step": step, "eval": {k: round(v, 5) for k, v in metrics.items()},
               "chain": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in chain.items()},
               **({"compare15": compare} if compare else {}),
               "score": round(sc, 5), "best": round(best, 5), "stale": stale, "free_gb": round(disk_free_gb(run), 1)})
        w = {**weights(), "eval": metrics, "chain": chain, "score": sc, **({"compare15": compare} if compare else {})}
        torch.save(w, run / f"step{step}.pt")
        if sc <= best:
            torch.save(w, run / "best.pt")
        save_last()
        if not math.isfinite(sc) or not chain["finite"]:
            return "non-finite validation"
        if len(history) >= 3 and sc > 2 * min(history):
            return f"divergence: score {sc:.4f} > 2x best {min(history):.4f}"
        if compare and args.target_dtw is not None and compare["dtw"] <= args.target_dtw:
            return f"target: 15-item full arm {compare['dtw']:.3f} <= {args.target_dtw} at step {step}"
        if stale >= args.patience:
            return f"no {'new best' if compare else '0.5% validation improvement'} in {stale} evaluations"
        if args.gate:
            gate_step, gate_dtw = args.gate.split(":")
            if step >= int(gate_step) and chain["dtw"] > float(gate_dtw):
                return f"gate: chain DTW {chain['dtw']:.3f} > {gate_dtw} at step {step}"
        if disk_free_gb(run) < args.min_free_gb:
            return f"disk free {disk_free_gb(run):.1f} GB < {args.min_free_gb}"
        return None

    start = time.time()
    stop_reason = run_eval() if args.init else None  # the v1 starting point, measured here on this device
    running, misses, done_here = {}, 0, 0
    while stop_reason is None and step < args.steps:
        indices = rng.sample(usable, args.utterances)
        use_student = rng.random() < args.student_share
        with ExactFP32():
            items = train.inputs(indices, use_student)
        targets = []
        for i, it in zip(indices, items):
            cached = cache.get(i, use_student, it["frames"]) if cache else None
            if cached is None:
                misses += 1
                with ExactFP32():
                    cached = pitch_teacher(model, it)
            targets.append(cached)

        f0_loss = n_loss = 0
        preds = []
        for it, (f0_t, n_t) in zip(items, targets):
            f0_s, n_s = pitch(it["en"], it["style"][:, 128:])
            preds.append((f0_s, n_s))
            a, b = pitch_loss(f0_s, n_s, f0_t, n_t)
            f0_loss, n_loss = f0_loss + a / len(items), n_loss + b / len(items)
        losses = dict(f0=f0_loss, energy=n_loss)
        total = f0_loss + n_loss

        ramp = min(1.0, max(0.0, (step - args.adv_start) / args.adv_ramp))
        if args.pitch_loss_only:
            win = None
        elif args.freeze_vocoder:  # teacher on its F0/N vs frozen vocoder on the student's F0/N, same window + noise
            crop, at, noise = rng.choice(crops), rng.random(), rng.randint(0, 2**31 - 1)
            win = crop_windows(items, targets, crop, random.Random(at))
            win_s = crop_windows(items, preds, crop, random.Random(at))
            if win is not None:
                asr, f0, n, style = win
                with ExactFP32():
                    torch.manual_seed(noise)
                    lt = teacher_logits(model, asr, f0, n, style[:, :128], sine_spectrum(model, f0))
                    torch.manual_seed(noise)
                    har_s = sine_spectrum(model, win_s[1].detach())  # Kokoro's sine source runs under no_grad
                ls = vocoder.logits(asr, win_s[1], win_s[2], style[:, :128], har_s)
                v = vocoder_losses(model, ls, lt, logmel, mrstft)
                losses.update(v)
                total = total + v["mel"] + v["mrstft"] + 0.5 * v["logmag"] + v["complex"]
            win = None
        else:
            win = crop_windows(items, targets, rng.choice(crops), rng)
        loss_d = None
        if win is not None:
            asr, f0, n, style = win
            torch.manual_seed(rng.randint(0, 2**31 - 1))
            with ExactFP32():
                har = sine_spectrum(model, f0)
                lt = teacher_logits(model, asr, f0, n, style[:, :128], har)
                audio_t = head_audio(model, lt)[2]
            ls = vocoder.logits(asr, f0, n, style[:, :128], har)
            v = vocoder_losses(model, ls, lt, logmel, mrstft)
            losses.update(v)
            total = total + v["mel"] + v["mrstft"] + 0.5 * v["logmag"] + v["complex"]
            audio_s = head_audio(model, ls)[2]

            if step >= args.disc_start:
                loss_d = d_loss(disc(audio_t), disc(audio_s.detach()))
                optimizer_d.zero_grad(set_to_none=True)
                loss_d.backward()
                bad = [k for k, q in disc.named_parameters() if q.grad is not None and not torch.isfinite(q.grad).all()]
                if not torch.isfinite(loss_d) or bad:
                    stop_reason = f"non-finite discriminator at step {step}: {float(loss_d)} {bad[:4]}"
                    break
                grad_d = torch.nn.utils.clip_grad_norm_(disc.parameters(), 10.0)
                optimizer_d.step()
                losses["disc"] = loss_d
                running["grad_d"] = running.get("grad_d", 0.0) + float(grad_d)
            if ramp > 0:
                with torch.no_grad():
                    real = disc(audio_t)
                adv, fm = g_losses(real, disc(audio_s))
                losses.update(adv=adv, fm=fm)
                total = total + ramp * (args.adv_weight * adv + args.fm_weight * fm)

        if not torch.isfinite(total):
            stop_reason = f"non-finite loss at step {step}: {({k: float(x) for k, x in losses.items()})}"
            break
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        bad = [name for name, q in [*pitch.named_parameters(prefix="pitch"), *vocoder.named_parameters(prefix="vocoder")]
               if q.grad is not None and not torch.isfinite(q.grad).all()]
        if bad:
            stop_reason = f"non-finite gradient at step {step}: {bad[:8]} ({len(bad)} tensors)"
            break
        grad = torch.nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
        scheduler.step()
        scheduler_d.step()
        step += 1
        done_here += 1
        for k, x in losses.items():
            running[k] = running.get(k, 0.0) + float(x)
        running["grad"] = running.get("grad", 0.0) + float(grad)

        if step % 100 == 0 or (args.stop_after and done_here == args.stop_after):
            count = 100 if step % 100 == 0 else done_here % 100 or 100
            write({"step": step, "sec": round(time.time() - start), "lr": scheduler.get_last_lr()[0], "ramp": round(ramp, 3),
                   "cache_misses": misses, "gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
                   if torch.cuda.is_available() else None, **{k: round(x / count, 5) for k, x in running.items()}})
            running = {}
        if step % args.eval_every == 0 or step == args.steps:
            stop_reason = run_eval()
        if stop_reason is None and args.max_minutes and time.time() - start > 60 * args.max_minutes:
            stop_reason = (run_eval() if step % args.eval_every else None) or f"time limit {args.max_minutes} min"
        if args.stop_after and done_here >= args.stop_after and stop_reason is None:
            stop_reason = f"stop-after {args.stop_after} steps"
    save_last()  # newest valid state: every break above happens before an optimizer step
    write({"done": step, "sec": round(time.time() - start), "stop": stop_reason or "steps complete", "best": best,
           "cache_misses": misses})


if __name__ == "__main__":
    main()
