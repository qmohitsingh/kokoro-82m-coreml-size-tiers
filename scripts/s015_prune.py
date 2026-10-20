"""Prune Kokoro's fp32 decoder + generator to the S015 vocoder shape, then refit layer by layer.

    target shape: VocoderStudent(width 256, gen 256, asr 32, last 80), the v3-w80 shape
                  (19,648,868 bytes projected app at int8, measured on v3 step 22000)
    teacher -> student
      decoder   encode 514->1024        -> 514->256
                decode.0 (1090->1024)    -> decode.0
                decode.1 + decode.2      -> decode.1 (norm1 from decode.1, refit to decode.2's output)
                decode.3 (up, ->512)     -> decode.2 (up, ->256)
                asr_res 64               -> 32
      generator stages 256 / 128         -> 128 / 80
    channels kept per stream: activation std (train crops) x norm of the teacher weights that
    consume the channel, top-k
    init: every tensor sliced from the teacher (weight norm folded); per-channel tensors (AdaIN,
          instance norm, Snake alphas, biases, depthwise pool, noise_convs rows) stay exact
    refit, in forward order, on the student's own (already refitted) activations:
      every conv whose input is pruned -> ridge least squares to the teacher conv's output on the
                                          kept channels (conv_post: to the teacher's logits)
      ups.0 / ups.1                    -> Adam (MSE) to the teacher's ups output on kept channels
    data: train-split crops only (fit and held-out batches from different seeds); the rng 7
          comparison items and the hidden set are never read
    -> artifacts/s015/runs/<name>/init.pt (+ v3 pitch student, v3 discriminators) + prune.json

    python scripts/s015_prune.py --name v4-prune --pitch artifacts/s015/runs/v3-w80/step26000.pt \
        --disc artifacts/s015/runs/v3-w80/last.pt --reference artifacts/s015/runs/v3-w80/step26000.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s014_teacher as teacher  # noqa: E402
from s014_student import Student  # noqa: E402
from s014_train import Split  # noqa: E402
from s015_downstream import VocoderStudent  # noqa: E402
from s015_train import S014_STUDENT, teacher_logits  # noqa: E402
from s015_widen import batches, logmag_l1, teacher_tensors  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = ((0, 0, 0), (1, 1, 2), (2, 3, 3))  # student decode.i <- teacher norm1 from decode.j, rest from decode.k


class Stop(Exception):
    pass


def grab(module, name, use_output, fn):
    """Run fn() until the named submodule runs; return its input (or output)."""
    got = {}

    def hook(m, i, o):
        got["x"] = o if use_output else i[0]
        raise Stop

    h = module.get_submodule(name).register_forward_hook(hook)
    try:
        fn()
    except Stop:
        pass
    finally:
        h.remove()
    return got["x"]


def unfold(x, conv):
    k, d, p = conv.kernel_size[0], conv.dilation[0], conv.padding[0]
    cols = F.unfold(x[:, :, None], (1, k), dilation=(1, d), padding=(0, p))  # [B, C*k, T]
    return cols.transpose(1, 2).reshape(-1, cols.shape[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--pitch", type=Path, required=True, help="checkpoint whose pitch student is kept")
    parser.add_argument("--disc", type=Path, required=True, help="last.pt whose discriminators are kept")
    parser.add_argument("--reference", type=Path, help="S015 checkpoint scored on the same held-out batches")
    parser.add_argument("--fit-batches", type=int, default=6)
    parser.add_argument("--held-batches", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=1e-3, help="x mean diagonal of the normal matrix")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    run = ROOT / "artifacts/s015/runs" / args.name
    run.mkdir(parents=True, exist_ok=True)
    began = time.time()

    model = teacher.load_teacher("cpu")
    dec = model.decoder
    voice = teacher.load_voice()
    saved = torch.load(S014_STUDENT, map_location="cpu")
    s014 = Student(**saved["config"])
    s014.load_state_dict(saved["model"])
    s014.eval()
    split = Split("train")
    fit = batches(split, voice, s014, model, 303, args.fit_batches)
    held = batches(split, voice, s014, model, 404, args.held_batches)
    print(f"batches fit {len(fit)} held {len(held)} ({time.time() - began:.0f}s)", flush=True)

    tt = {**{f"generator.{k}": v for k, v in teacher_tensors(dec.generator).items()},
          **{k: v for k, v in teacher_tensors(dec).items() if not k.startswith("generator.")}}
    student = VocoderStudent(last=80)
    width, gen, asr = student.config["width"], student.config["gen"], student.config["asr"]
    g0, g1 = student.ups[0].out_channels, student.ups[1].out_channels

    # channel statistics on the teacher: std of each stream where its consumer reads it
    def teacher_run(data):
        return lambda: teacher_logits(model, *data)

    taps = {"E": ("decode.0", 0, 1024), "ASR": ("decode.0", 1024, 1088), "D0": ("decode.1", 0, 1024),
            "D2": ("decode.3", 0, 1024), "IE": ("encode.conv2", 0, 1024), "I0": ("decode.0.conv2", 0, 1024),
            "I2": ("decode.2.conv2", 0, 1024), "I3": ("decode.3.conv2", 0, 512), "D3": ("generator.ups.0", 0, 512),
            "G0": ("generator.ups.1", 0, 256), "G1": ("generator.conv_post", 0, 128)}
    std = {}
    for key, (name, a, b) in taps.items():
        xs = [grab(dec, name, False, teacher_run(d))[:, a:b] for d in fit]
        std[key] = torch.cat([x.transpose(0, 1).flatten(1) for x in xs], 1).std(1)
    col = lambda w, a, b: w[:, a:b].transpose(0, 1).flatten(1).norm(dim=1)  # per input channel
    both = lambda blk, a, b: (col(tt[f"{blk}.conv1.weight"], a, b) ** 2 + col(tt[f"{blk}.conv1x1.weight"], a, b) ** 2).sqrt()
    consumer = {
        "E": both("decode.0", 0, 1024), "D0": both("decode.1", 0, 1024), "D2": both("decode.3", 0, 1024),
        "ASR": sum(both(f"decode.{i}", 1024, 1088) ** 2 for i in range(4)).sqrt(),
        "IE": col(tt["encode.conv2.weight"], 0, 1024), "I0": col(tt["decode.0.conv2.weight"], 0, 1024),
        "I2": col(tt["decode.2.conv2.weight"], 0, 1024), "I3": col(tt["decode.3.conv2.weight"], 0, 512),
        "D3": tt["generator.ups.0.weight"].flatten(1).norm(dim=1),
        "G0": tt["generator.ups.1.weight"].flatten(1).norm(dim=1),
        "G1": col(tt["generator.conv_post.weight"], 0, 128)}
    keep_n = {"E": width, "D0": width, "D2": width, "IE": width, "I0": width, "I2": width, "I3": gen, "D3": gen,
              "ASR": asr, "G0": g0, "G1": g1}
    keep = {k: (std[k] * consumer[k]).argsort(descending=True)[:keep_n[k]].sort().values for k in keep_n}
    print("kept " + ", ".join(f"{k} {len(v)}/{len(std[k])}" for k, v in keep.items()), flush=True)

    # init: slice every tensor
    def cat_in(stream):  # teacher input channels of a decode block: [x, res, F0, N]
        return torch.cat([keep[stream], 1024 + keep["ASR"], torch.tensor([1088, 1089])])

    def put(sname, w):
        p = student.get_parameter(sname)
        assert p.shape == w.shape, (sname, p.shape, w.shape)
        p.copy_(w)

    def conv(sname, tname, out_idx, in_idx):
        w = tt[f"{tname}.weight"]
        w = w if out_idx is None else w[out_idx]
        put(f"{sname}.weight", w if in_idx is None else w[:, in_idx])
        if f"{tname}.bias" in tt:
            put(f"{sname}.bias", tt[f"{tname}.bias"] if out_idx is None else tt[f"{tname}.bias"][out_idx])

    def adain(sname, tname, idx):
        c = tt[f"{tname}.norm.weight"].shape[0]
        put(f"{sname}.norm.weight", tt[f"{tname}.norm.weight"][idx])
        put(f"{sname}.norm.bias", tt[f"{tname}.norm.bias"][idx])
        put(f"{sname}.fc.weight", tt[f"{tname}.fc.weight"][torch.cat([idx, c + idx])])
        put(f"{sname}.fc.bias", tt[f"{tname}.fc.bias"][torch.cat([idx, c + idx])])

    def snake(sname, tname, idx):
        for j in range(3):
            for which in ("convs1", "convs2"):
                conv(f"{sname}.{which}.{j}", f"{tname}.{which}.{j}", idx, idx)
            for which in ("adain1", "adain2"):
                adain(f"{sname}.{which}.{j}", f"{tname}.{which}.{j}", idx)
            for which in ("alpha1", "alpha2"):
                put(f"{sname}.{which}.{j}", tt[f"{tname}.{which}.{j}"][:, idx])

    conv("f0_conv", "F0_conv", None, None)
    conv("n_conv", "N_conv", None, None)
    conv("asr_res", "asr_res.0", keep["ASR"], None)
    adain("encode.norm1", "encode.norm1", torch.arange(514))
    conv("encode.conv1", "encode.conv1", keep["IE"], None)
    adain("encode.norm2", "encode.norm2", keep["IE"])
    conv("encode.conv2", "encode.conv2", keep["E"], keep["IE"])
    conv("encode.conv1x1", "encode.conv1x1", keep["E"], None)
    streams = {0: ("E", "I0", "D0"), 1: ("D0", "I2", "D2"), 2: ("D2", "I3", "D3")}
    for s, j, k in BLOCKS:
        src, mid, out = streams[s]
        adain(f"decode.{s}.norm1", f"decode.{j}.norm1", cat_in(src))
        conv(f"decode.{s}.conv1", f"decode.{k}.conv1", keep[mid], cat_in(src))
        adain(f"decode.{s}.norm2", f"decode.{k}.norm2", keep[mid])
        conv(f"decode.{s}.conv2", f"decode.{k}.conv2", keep[out], keep[mid])
        conv(f"decode.{s}.conv1x1", f"decode.{k}.conv1x1", keep[out], cat_in(src))
    student.decode[2].pool.weight.copy_(tt["decode.3.pool.weight"][cat_in("D2")])
    student.decode[2].pool.bias.copy_(tt["decode.3.pool.bias"][cat_in("D2")])
    for i, (src, g) in enumerate((("D3", "G0"), ("G0", "G1"))):
        w = tt[f"generator.ups.{i}.weight"][keep[src]][:, keep[g]]
        put(f"ups.{i}.weight", w)
        put(f"ups.{i}.bias", tt[f"generator.ups.{i}.bias"][keep[g]])
        conv(f"noise_convs.{i}", f"generator.noise_convs.{i}", keep[g], None)
        snake(f"noise_res.{i}", f"generator.noise_res.{i}", keep[g])
        for r in range(3):
            snake(f"resblocks.{3 * i + r}", f"generator.resblocks.{3 * i + r}", keep[g])
    conv("conv_post", "generator.conv_post", None, keep["G1"])
    student.eval()

    def held_l1(vocoder):
        return float(np.mean([logmag_l1(vocoder.logits(*d), teacher_logits(model, *d)) for d in held]))

    scores = {"slice_only": held_l1(student)}
    print(f"held-out log-mag L1, sliced: {scores['slice_only']:.4f} ({time.time() - began:.0f}s)", flush=True)

    # refit in forward order
    plan = [("encode.conv2", "encode.conv2", "E")]
    for s, j, k in BLOCKS:
        _, mid, out = streams[s]
        plan += [(f"decode.{s}.conv1", f"decode.{k}.conv1", mid), (f"decode.{s}.conv2", f"decode.{k}.conv2", out),
                 (f"decode.{s}.conv1x1", f"decode.{k}.conv1x1", out)]
    for i, g in enumerate(("G0", "G1")):
        names = [f"noise_res.{i}"] + [f"resblocks.{3 * i + r}" for r in range(3)]
        plan.append((f"ups.{i}", f"generator.ups.{i}", g))
        for blk in names:
            for j in range(3):
                plan += [(f"{blk}.convs1.{j}", f"generator.{blk}.convs1.{j}", g),
                         (f"{blk}.convs2.{j}", f"generator.{blk}.convs2.{j}", g)]
    plan.append(("conv_post", "generator.conv_post", None))

    fits = {}
    for sname, tname, stream in plan:
        sub = student.get_submodule(sname)
        out_idx = slice(None) if stream is None else keep[stream]
        xs = [grab(student, sname, False, lambda d=d: student.logits(*d)) for d in fit]
        if sname.startswith("decode.") and sname.endswith("conv1x1"):  # shortcut fits the block's remainder
            blk = sname.rsplit(".", 1)[0]
            ys = [grab(dec, tname.rsplit(".", 1)[0], True, teacher_run(d))[:, out_idx] * 2 ** 0.5
                  - grab(student, f"{blk}.conv2", True, lambda d=d: student.logits(*d)) for d in fit]
        else:
            ys = [grab(dec, tname, True, teacher_run(d))[:, out_idx] for d in fit]
        saved_sub = {k: v.clone() for k, v in sub.state_dict().items()}
        before = float(sum(F.mse_loss(sub(x), y) for x, y in zip(xs, ys)) / len(xs))
        if isinstance(sub, torch.nn.ConvTranspose1d):
            with torch.enable_grad():
                opt = torch.optim.Adam(sub.parameters(), lr=1e-3)
                for _ in range(300):
                    loss = sum(F.mse_loss(sub(x), y) for x, y in zip(xs, ys)) / len(xs)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        else:
            n, b = sub.weight[0].numel(), int(sub.bias is not None)
            gram = torch.zeros(n + b, n + b, dtype=torch.float64)
            cross = torch.zeros(n + b, sub.out_channels, dtype=torch.float64)
            for x, y in zip(xs, ys):
                a = unfold(x, sub).double()
                if b:
                    a = torch.cat([a, torch.ones(a.shape[0], 1, dtype=torch.float64)], 1)
                gram += a.T @ a
                cross += a.T @ y.transpose(1, 2).reshape(-1, y.shape[1]).double()
            reg = args.ridge * gram.diagonal()[:n].mean() * torch.eye(n + b, dtype=torch.float64)
            if b:
                reg[-1, -1] = 0
            sol = torch.linalg.solve(gram + reg, cross).float()  # [n + b, out]
            sub.weight.copy_(sol[:n].T.reshape(sub.weight.shape))
            if b:
                sub.bias.copy_(sol[-1])
        after = float(sum(F.mse_loss(sub(x), y) for x, y in zip(xs, ys)) / len(xs))
        if after > before:  # keep the sliced weights
            sub.load_state_dict(saved_sub)
            after = before
        var = float(sum(y.var() for y in ys) / len(ys))
        fits[sname] = dict(r2_sliced=1 - before / var, r2_refit=1 - after / var)
        print(f"{sname:28s} R2 sliced {1 - before / var:+.3f} refit {1 - after / var:+.3f} "
              f"({time.time() - began:.0f}s)", flush=True)

    scores["refit"] = held_l1(student)
    if args.reference:
        ref = torch.load(args.reference, map_location="cpu")
        v = VocoderStudent(**ref["vocoder_config"])
        v.load_state_dict(ref["vocoder"])
        scores["reference"] = held_l1(v.eval())
    print(f"held-out log-mag L1 {json.dumps({k: round(x, 4) for k, x in scores.items()})}", flush=True)

    pitch_state = torch.load(args.pitch, map_location="cpu")
    disc_state = torch.load(args.disc, map_location="cpu")
    meta = dict(blocks=BLOCKS, kept={k: v.tolist() for k, v in keep.items()}, fits=fits, held_logmag_l1=scores,
                pitch_from=str(args.pitch), disc_from=str(args.disc), reference=str(args.reference),
                fit_batches=len(fit), held_batches=len(held), ridge=args.ridge, seconds=round(time.time() - began))
    torch.save({"pitch": pitch_state["pitch"], "pitch_config": pitch_state.get("pitch_config"),
                "vocoder": student.state_dict(), "vocoder_config": student.config,
                "disc": disc_state["disc"], "optimizer_d": disc_state["optimizer_d"], "step": 0, "prune": meta},
               run / "init.pt")
    (run / "prune.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {run / 'init.pt'} config {student.config} ({time.time() - began:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
