"""Distil Kokoro's fp32 pitch, decoder and generator into the S015 downstream students.

Targets always come from the original fp32 Kokoro-82M teacher, run on the same
inputs the students see. Never from S014's compressed Core ML audio.

    batch of train utterances (books + templates; never the hidden test or frozen chunks)
      inputs: teacher front-end cache (fp32 teacher durations, int8-stored features)
              or, with --student-share, the frozen S014 PyTorch student's own outputs
      full utterance: aligned hidden -> teacher F0Ntrain -> f0, n  (fp32, no grad)
                      PitchStudent on the same full utterance -> L1 to teacher f0, n
      crop window:    sine source from teacher f0 (one seeded noise draw, shared)
                      teacher decoder + generator -> head logits, audio  (no grad)
                      VocoderStudent(text, teacher f0, n, style, same spectrum)
                      -> log-mel + multi-res STFT + head log-magnitude + complex head
    every --eval-every steps: fixed validation windows from the holdout books
    -> artifacts/s015/runs/<name>/{last.pt, stepN.pt, log.jsonl}
    stops on NaN, divergence, low disk, or no validation improvement

    .venv-convert/bin/python scripts/s015_train.py --name v1
"""

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s014_teacher as teacher  # noqa: E402
from s014_student import Student  # noqa: E402
from s014_train import LogMel, Split  # noqa: E402
from s015_downstream import PitchStudent, VocoderStudent, parameter_count  # noqa: E402
from s015_istft import RealISTFT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
S014_STUDENT = ROOT / "artifacts/s014/runs/w256-b/step10000.pt"
LOG_FLOOR = math.log(1e-5)


def power(x, n_fft, hop, window):
    """|STFT|^2 from real and imaginary parts; complex abs() has a NaN gradient at 0."""
    z = torch.view_as_real(torch.stft(x, n_fft, hop, window=window, return_complex=True))
    return z.pow(2).sum(-1)


class SafeLogMel(LogMel):
    """S014's multi-resolution log-mel with a finite gradient at zero bins."""

    def forward(self, a, b):
        loss = 0
        for (n_fft, hop), fb, window in zip(self.sizes, self.filters, self.windows):
            ma = torch.log(fb @ power(a, n_fft, hop, window) + 1e-5)
            mb = torch.log(fb @ power(b, n_fft, hop, window) + 1e-5)
            loss = loss + (ma - mb).abs().mean()
        return loss / len(self.sizes)


class MultiResSTFT(torch.nn.Module):
    """Spectral convergence + log-magnitude L1, five resolutions down to 128-point FFTs."""

    def __init__(self, device, sizes=((2048, 512), (1024, 256), (512, 128), (256, 64), (128, 32))):
        super().__init__()
        self.sizes = sizes
        self.windows = [torch.hann_window(n, device=device) for n, _ in sizes]

    def forward(self, a, b):
        loss = 0
        for (n_fft, hop), window in zip(self.sizes, self.windows):
            sa = (power(a, n_fft, hop, window) + 1e-10).sqrt()
            sb = (power(b, n_fft, hop, window) + 1e-10).sqrt()
            sc = torch.linalg.norm(sa - sb) / torch.linalg.norm(sb).clamp(min=1e-7)
            loss = loss + sc + (sa.log() - sb.log()).abs().mean()
        return loss / len(self.sizes)


@torch.no_grad()
def sine_spectrum(model, f0):
    """Teacher's harmonic source -> [B, 22, 120F + 1]. Uses the current torch RNG."""
    gen = model.decoder.generator
    f0u = gen.f0_upsamp(f0[:, None]).transpose(1, 2)
    har, _, _ = gen.m_source(f0u)
    mag, phase = gen.stft.transform(har.transpose(1, 2).squeeze(1))
    return torch.cat([mag, phase], 1)


@torch.no_grad()
def teacher_logits(model, asr, f0, n, s, har):
    """fp32 Kokoro decoder + generator up to conv_post, with a given source spectrum."""
    dec = model.decoder
    F0 = dec.F0_conv(f0[:, None])
    N = dec.N_conv(n[:, None])
    x = dec.encode(torch.cat([asr, F0, N], 1), s)
    res = dec.asr_res(asr)
    up = True
    for block in dec.decode:
        if up:
            x = torch.cat([x, res, F0, N], 1)
        x = block(x, s)
        if block.upsample_type != "none":
            up = False
    gen = dec.generator
    for i in range(gen.num_upsamples):
        x = F.leaky_relu(x, 0.1)
        source = gen.noise_res[i](gen.noise_convs[i](har), s)
        x = gen.ups[i](x)
        if i == gen.num_upsamples - 1:
            x = gen.reflection_pad(x)
        x = x + source
        xs = None
        for j in range(gen.num_kernels):
            y = gen.resblocks[i * gen.num_kernels + j](x, s)
            xs = y if xs is None else xs + y
        x = xs / gen.num_kernels
    return gen.conv_post(F.leaky_relu(x))


ISTFT = RealISTFT()


def head_audio(model, logits):
    mag = torch.exp(logits[:, :11])
    phase = torch.sin(logits[:, 11:])
    return mag, phase, ISTFT.to(logits.device)(mag, phase)


def vocoder_losses(model, logits_s, logits_t, logmel, mrstft):
    mag_s, ph_s, audio_s = head_audio(model, logits_s)
    mag_t, ph_t, audio_t = head_audio(model, logits_t)
    logmag = (logits_s[:, :11].clamp(min=LOG_FLOOR) - logits_t[:, :11].clamp(min=LOG_FLOOR)).abs().mean()
    scale = mag_t.mean().clamp(min=1e-6)
    complex_ = ((mag_s * torch.cos(ph_s) - mag_t * torch.cos(ph_t)).abs()
                + (mag_s * torch.sin(ph_s) - mag_t * torch.sin(ph_t)).abs()).mean() / scale
    # spectral losses on CPU for MPS only: torch.stft on MPS gave intermittent NaN gradients
    spec_s, spec_t = (audio_s.cpu(), audio_t.cpu()) if audio_s.device.type == "mps" else (audio_s, audio_t)
    return dict(mel=logmel(spec_s, spec_t).to(logits_s.device), mrstft=mrstft(spec_s, spec_t).to(logits_s.device),
                logmag=logmag, complex=complex_)


class Batcher:
    """Utterances -> full-utterance pitch items and same-length vocoder crops."""

    def __init__(self, split: Split, voice, device, s014=None):
        self.split, self.voice, self.device, self.s014 = split, voice, device, s014

    @torch.no_grad()
    def inputs(self, indices, use_student: bool):
        batch = self.split.batch(indices, self.voice, self.device)
        if use_student:
            hidden, logits, text = self.s014(batch["ids"], batch["style"], batch["keep"][..., None])
            hidden = hidden[..., :512]
            text = text.transpose(1, 2)
            frames = torch.round(torch.sigmoid(logits).sum(-1)).clamp(min=1)
        else:
            hidden, text = batch["hidden"], batch["text"]
            frames = torch.round(batch["duration"]).clamp(min=1)
        items = []
        for j in range(len(indices)):
            n = int(batch["lengths"][j])
            fr = frames[j, :n].long()
            aln = teacher.alignment(fr, n, int(fr.sum()))
            style = batch["style"][j : j + 1]
            en = torch.cat([hidden[j, :n].T @ aln, style[0, 128:, None].expand(-1, aln.shape[1])], 0)
            items.append(dict(en=en[None], asr=(text[j, :n].T @ aln)[None], style=style, frames=aln.shape[1]))
        return items


def pitch_teacher(model, item):
    with torch.no_grad():
        return model.predictor.F0Ntrain(item["en"], item["style"][:, 128:])


def crop_windows(items, targets, crop, rng):
    """One window of `crop` frames per long-enough utterance, stacked."""
    asr, f0, n, style = [], [], [], []
    for item, (f0_t, n_t) in zip(items, targets):
        if item["frames"] < crop:
            continue
        a = rng.randint(0, item["frames"] - crop)
        asr.append(item["asr"][:, :, a : a + crop])
        f0.append(f0_t[:, 2 * a : 2 * (a + crop)])
        n.append(n_t[:, 2 * a : 2 * (a + crop)])
        style.append(item["style"])
    if not asr:
        return None
    return torch.cat(asr), torch.cat(f0), torch.cat(n), torch.cat(style)


def pitch_loss(f0_s, n_s, f0_t, n_t):
    return (f0_s - f0_t).abs().mean() / 100, (n_s - n_t).abs().mean()


def disk_free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


@torch.no_grad()
def evaluate(model, pitch, vocoder, val: Batcher, logmel, mrstft, crop, seeds=8) -> dict:
    """Fixed holdout windows, half teacher-cache inputs and half S014-student inputs."""
    pitch.eval(), vocoder.eval()
    rng = random.Random(123)
    order = [i for i in range(len(val.split)) if val.split.lengths[i] >= 60]
    rng.shuffle(order)
    sums, count = {}, 0
    for k in range(seeds):
        indices = order[k * 8 : (k + 1) * 8]
        for use_student in (False, True):
            items = val.inputs(indices, use_student)
            targets = [pitch_teacher(model, it) for it in items]
            out = {}
            f0_l = n_l = 0
            for it, (f0_t, n_t) in zip(items, targets):
                f0_s, n_s = pitch(it["en"], it["style"][:, 128:])
                a, b = pitch_loss(f0_s, n_s, f0_t, n_t)
                f0_l, n_l = f0_l + float(a) / len(items), n_l + float(b) / len(items)
            out.update(f0=f0_l, energy=n_l)
            win = crop_windows(items, targets, crop, random.Random(k))
            if win is not None:
                asr, f0, n, style = win
                torch.manual_seed(1000 + k)
                har = sine_spectrum(model, f0)
                lt = teacher_logits(model, asr, f0, n, style[:, :128], har)
                ls = vocoder.logits(asr, f0, n, style[:, :128], har)
                out.update({k2: float(v) for k2, v in vocoder_losses(model, ls, lt, logmel, mrstft).items()})
            tag = "s014" if use_student else "teacher"
            for key, v in out.items():
                sums[f"{tag}_{key}"] = sums.get(f"{tag}_{key}", 0.0) + v
        count += 1
    pitch.train(), vocoder.train()
    return {k: v / count for k, v in sums.items()}


def score(metrics: dict) -> float:
    """Lower is better; audio terms dominate, pitch terms keep the pitch model honest."""
    total = 0.0
    for tag in ("teacher", "s014"):
        total += metrics[f"{tag}_mel"] + metrics[f"{tag}_mrstft"] + 0.5 * metrics[f"{tag}_logmag"]
        total += metrics[f"{tag}_f0"] + metrics[f"{tag}_energy"]
    return total / 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--utterances", type=int, default=8, help="utterances per step")
    parser.add_argument("--crops", default="80,96,112,128", help="vocoder window lengths in frames, one per step")
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--student-share", type=float, default=0.5, help="share of steps on S014 student inputs")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=6, help="evals without 0.5%% improvement before stopping")
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--max-seconds", type=float, default=0, help="stop cleanly after this wall time")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--debug-grads", action="store_true")
    args = parser.parse_args()

    run = ROOT / "artifacts/s015/runs" / args.name
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(args), default=str, indent=2))
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

    pitch, vocoder = PitchStudent().to(args.device), VocoderStudent().to(args.device)
    params = [*pitch.parameters(), *vocoder.parameters()]
    print(f"pitch {parameter_count(pitch)} vocoder {parameter_count(vocoder)}", flush=True)
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.8, 0.99), weight_decay=0.0)
    schedule = lambda step: min(1, (step + 1) / args.warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(step, args.steps) / args.steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    logmel, mrstft = SafeLogMel("cpu"), MultiResSTFT("cpu")

    step, best, stale, history = 0, float("inf"), 0, []
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        pitch.load_state_dict(state["pitch"])
        vocoder.load_state_dict(state["vocoder"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, best, stale, history = state["step"], state["best"], state["stale"], state["history"]
        rng.seed(step)
        print(f"resumed at step {step}", flush=True)

    log = open(run / "log.jsonl", "a")
    write = lambda row: (print(json.dumps(row), flush=True), log.write(json.dumps(row) + "\n"), log.flush())
    start = time.time()
    running = {}
    stop_reason = None
    while step < args.steps:
        indices = rng.sample(usable, args.utterances)
        use_student = rng.random() < args.student_share
        items = train.inputs(indices, use_student)
        targets = [pitch_teacher(model, it) for it in items]

        f0_loss = n_loss = 0
        for it, (f0_t, n_t) in zip(items, targets):
            f0_s, n_s = pitch(it["en"], it["style"][:, 128:])
            a, b = pitch_loss(f0_s, n_s, f0_t, n_t)
            f0_loss, n_loss = f0_loss + a / len(items), n_loss + b / len(items)
        losses = dict(f0=f0_loss, energy=n_loss)
        total = f0_loss + n_loss

        win = crop_windows(items, targets, rng.choice(crops), rng)
        if win is not None:
            asr, f0, n, style = win
            torch.manual_seed(rng.randint(0, 2**31 - 1))
            har = sine_spectrum(model, f0)
            lt = teacher_logits(model, asr, f0, n, style[:, :128], har)
            ls = vocoder.logits(asr, f0, n, style[:, :128], har)
            if not (torch.isfinite(lt).all() and torch.isfinite(ls).all()):
                print(json.dumps({"nonfinite_step": step, "har": bool(torch.isfinite(har).all()),
                                  "asr": bool(torch.isfinite(asr).all()), "f0": bool(torch.isfinite(f0).all()),
                                  "teacher": bool(torch.isfinite(lt).all()), "student": bool(torch.isfinite(ls).all()),
                                  "crop": asr.shape[-1], "f0_range": [float(f0.min()), float(f0.max())]}), flush=True)
            v = vocoder_losses(model, ls, lt, logmel, mrstft)
            if args.debug_grads:
                for key, term in v.items():
                    g = torch.autograd.grad(term, ls, retain_graph=True)[0]
                    if not torch.isfinite(g).all():
                        idx = (~torch.isfinite(g)).nonzero()[:3].tolist()
                        print(json.dumps({"step": step, "bad_term": key, "where": idx, "len": ls.shape[-1]}), flush=True)
            losses.update(v)
            total = total + v["mel"] + v["mrstft"] + 0.5 * v["logmag"] + v["complex"]

        if not torch.isfinite(total):
            bad = {k: float(v) for k, v in losses.items()}
            stop_reason = f"non-finite loss at step {step}: {bad} student_inputs={use_student}"
            break
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        bad = [name for name, q in [*pitch.named_parameters(prefix="pitch"), *vocoder.named_parameters(prefix="vocoder")]
               if q.grad is not None and not torch.isfinite(q.grad).all()]
        if bad:
            stop_reason = f"non-finite gradient at step {step}: {bad[:8]} ({len(bad)} tensors) student_inputs={use_student}"
            break
        grad = torch.nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
        scheduler.step()
        step += 1
        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + float(v)
        running["grad"] = running.get("grad", 0.0) + float(grad)

        if step % 100 == 0:
            write({"step": step, "sec": round(time.time() - start), "lr": scheduler.get_last_lr()[0],
                   **{k: round(v / 100, 5) for k, v in running.items()}})
            running = {}
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, pitch, vocoder, val, logmel, mrstft, crop=96)
            sc = score(metrics)
            history.append(sc)
            improved = sc < best * 0.995
            if sc < best:
                best = sc
            stale = 0 if improved else stale + 1
            write({"step": step, "eval": {k: round(v, 5) for k, v in metrics.items()}, "score": round(sc, 5),
                   "best": round(best, 5), "stale": stale, "free_gb": round(disk_free_gb(run), 1)})
            weights = {"pitch": pitch.state_dict(), "vocoder": vocoder.state_dict(),
                       "pitch_config": pitch.config, "vocoder_config": vocoder.config, "step": step, "eval": metrics}
            torch.save(weights, run / f"step{step}.pt")
            if sc <= best:
                torch.save(weights, run / "best.pt")
            torch.save({**weights, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "best": best, "stale": stale, "history": history}, run / "last.pt")
            if not math.isfinite(sc):
                stop_reason = "non-finite validation"
            elif len(history) >= 3 and sc > 2 * min(history):
                stop_reason = f"divergence: score {sc:.4f} > 2x best {min(history):.4f}"
            elif stale >= args.patience:
                stop_reason = f"no 0.5% validation improvement in {stale} evaluations"
            elif disk_free_gb(run) < args.min_free_gb:
                stop_reason = f"disk free {disk_free_gb(run):.1f} GB < {args.min_free_gb}"
            if stop_reason:
                break
        if args.max_seconds and time.time() - start > args.max_seconds:
            stop_reason = f"wall time {args.max_seconds:.0f}s reached"
            break
    write({"done": step, "sec": round(time.time() - start), "stop": stop_reason or "steps complete", "best": best})


if __name__ == "__main__":
    main()
