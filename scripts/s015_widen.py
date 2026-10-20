"""Widen the S015 vocoder's final generator stage from 64 to --last channels, warm-started.

    v2 checkpoint -> every tensor outside the final stage copied unchanged (decoder body,
                     stage 0, pitch student, discriminators + their optimizer)
    final stage (ups.1, noise_convs.1, noise_res.1, resblocks.3-5, conv_post):
      channels 0..63   v2's trained channels, unchanged
      channels 64..    Kokoro's final-stage channels (128), strongest first, skipping any whose
                       activation correlates above --max-corr with a v2 channel or an already
                       picked one; strength = activation std x norm of its conv_post column
        per-channel tensors (biases, AdaIN, Snake alphas, noise_convs rows) from the teacher;
        new x new blocks of the resblock convs from the teacher, old <-> new blocks zero, so
        v2's channels are untouched by the new ones;
        ups.1 new outputs fitted (Adam, MSE) to the teacher's ups.1 output on those channels,
        since the teacher's stage-0 inputs (256) do not match v2's (128);
        conv_post new columns: best held-out log-magnitude L1 among zero (= v2 exactly),
        teacher columns x {0.25, 0.5, 1}, and a ridge fit to the teacher-minus-v2 residual
    data: train-split crops only (fit and held-out batches from different seeds), teacher and
          S014-student inputs as in training; the rng 7 comparison items and the hidden set
          are never read
    -> artifacts/s015/runs/<name>/init.pt (for s015_train_gan.py --init) + graft.json

    python scripts/s015_widen.py --name v3-w80 --last 80 --init artifacts/s015/runs/v2-a100/last.pt
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s014_teacher as teacher  # noqa: E402
from s014_student import Student  # noqa: E402
from s014_train import Split  # noqa: E402
from s015_downstream import VocoderStudent  # noqa: E402
from s015_train import LOG_FLOOR, S014_STUDENT, Batcher, crop_windows, pitch_teacher, sine_spectrum, teacher_logits  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FINAL = ("ups.1.", "noise_convs.1.", "noise_res.1.", "resblocks.3.", "resblocks.4.", "resblocks.5.", "conv_post.")


def teacher_tensors(gen):
    """Generator state with weight norm folded in: name -> effective tensor."""
    raw, out = gen.state_dict(), {}
    for k, v in raw.items():
        if k.endswith("weight_g"):
            out[k[:-2]] = torch._weight_norm(raw[k[:-1] + "v"], v, 0)
        elif not k.endswith("weight_v"):
            out[k] = v
    return out


def batches(split, voice, s014, model, seed, count, utterances=8, crop=96):
    rng = random.Random(seed)
    usable = [i for i in range(len(split)) if split.lengths[i] >= 40]
    batcher = Batcher(split, voice, "cpu", s014)
    out = []
    while len(out) < count:
        indices = rng.sample(usable, utterances)
        items = batcher.inputs(indices, use_student=len(out) % 2 == 1)
        win = crop_windows(items, [pitch_teacher(model, it) for it in items], crop, rng)
        if win is None:
            continue
        asr, f0, n, style = win
        torch.manual_seed(rng.randint(0, 2**31 - 1))
        out.append((asr, f0, n, style[:, :128], sine_spectrum(model, f0)))
    return out


def capture(module, names, fn):
    """Run fn() and return {name: (input, output)} of the named submodules."""
    got, hooks = {}, []
    for name in names:
        sub = module.get_submodule(name)
        hooks.append(sub.register_forward_hook(lambda m, i, o, name=name: got.__setitem__(name, (i[0], o))))
    try:
        result = fn()
    finally:
        for h in hooks:
            h.remove()
    return result, got


def flat(x):  # [B, C, T] -> [C, B*T]
    return x.transpose(0, 1).reshape(x.shape[1], -1)


def logmag_l1(ls, lt):
    return float((ls[:, :11].clamp(min=LOG_FLOOR) - lt[:, :11].clamp(min=LOG_FLOOR)).abs().mean())


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--init", type=Path, required=True, help="v2 last.pt")
    parser.add_argument("--last", type=int, default=80)
    parser.add_argument("--max-corr", type=float, default=0.5)
    parser.add_argument("--fit-batches", type=int, default=8)
    parser.add_argument("--held-batches", type=int, default=4)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    run = ROOT / "artifacts/s015/runs" / args.name
    run.mkdir(parents=True, exist_ok=True)

    model = teacher.load_teacher("cpu")
    gen = model.decoder.generator
    voice = teacher.load_voice()
    saved = torch.load(S014_STUDENT, map_location="cpu")
    s014 = Student(**saved["config"])
    s014.load_state_dict(saved["model"])
    s014.eval()
    state = torch.load(args.init, map_location="cpu")
    old = VocoderStudent(**state["vocoder_config"])
    old.load_state_dict(state["vocoder"])
    old.eval()
    base = old.ups[-1].out_channels
    extra = args.last - base
    assert extra > 0

    split = Split("train")
    fit = batches(split, voice, s014, model, 101, args.fit_batches)
    held = batches(split, voice, s014, model, 202, args.held_batches)
    print(f"batches fit {len(fit)} held {len(held)}", flush=True)

    def run_both(data):
        rows = []
        for asr, f0, n, s, har in data:
            lt, t = capture(gen, ["ups.1", "conv_post"], lambda: teacher_logits(model, asr, f0, n, s, har))
            ls, v = capture(old, ["ups.1", "conv_post"], lambda: old.logits(asr, f0, n, s, har))
            rows.append(dict(lt=lt, ls=ls, t_ups=t["ups.1"][1], t_post=t["conv_post"][0],
                             s_ups_in=v["ups.1"][0], s_post=v["conv_post"][0]))
        return rows

    fit_rows, held_rows = run_both(fit), run_both(held)

    # rank teacher final-stage channels, skip duplicates of v2's channels
    T = torch.cat([flat(r["t_post"]) for r in fit_rows], 1)
    S = torch.cat([flat(r["s_post"]) for r in fit_rows], 1)
    tt = teacher_tensors(gen)
    z = lambda x: (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-8)
    corr_old = (z(T) @ z(S).T / T.shape[1]).abs()  # [128, base]
    corr_tt = (z(T) @ z(T).T / T.shape[1]).abs()
    strength = T.std(1) * tt["conv_post.weight"].transpose(0, 1).flatten(1).norm(dim=1)
    picked, skipped = [], []
    for t in strength.argsort(descending=True).tolist():
        if len(picked) == extra:
            break
        if corr_old[t].max() > args.max_corr or any(corr_tt[t, p] > args.max_corr for p in picked):
            skipped.append(t)
            continue
        picked.append(t)
    assert len(picked) == extra, f"only {len(picked)} channels under max-corr {args.max_corr}"
    idx = torch.tensor(picked)
    print(f"picked {picked}\nskipped {len(skipped)} stronger duplicates; max |corr| to v2 of picked "
          f"{corr_old[idx].max(1).values.numpy().round(2).tolist()}", flush=True)

    # graft
    new = VocoderStudent(**{**state["vocoder_config"], "last": args.last})
    ns, os_ = new.state_dict(), old.state_dict()
    for k in ns:
        if not k.startswith(FINAL):
            ns[k].copy_(os_[k])
            continue
        o, t, w = os_[k], tt[k], ns[k]
        w.zero_()
        if ".fc." in k:  # AdaIN: gamma rows [0, C), beta rows [C, 2C)
            for half in range(2):
                w[half * args.last:half * args.last + base] = o[half * base:(half + 1) * base]
                w[half * args.last + base:(half + 1) * args.last] = t[half * 128 + idx]
            continue
        dims = [d for d in range(w.dim()) if w.shape[d] == args.last and o.shape[d] == base]
        w[tuple(slice(0, base) if d in dims else slice(None) for d in range(w.dim()))] = o
        sub = t
        for d in dims:
            sub = sub.index_select(d, idx)
        if k.startswith("ups.1.") or k.startswith("conv_post."):
            continue  # fitted below
        w[tuple(slice(base, None) if d in dims else slice(None) for d in range(w.dim()))] = sub
    new.load_state_dict(ns)
    new.eval()

    # ups.1 new outputs: fit to the teacher's ups.1 output on the picked channels
    up = torch.nn.ConvTranspose1d(new.ups[-1].in_channels, extra, new.ups[-1].kernel_size, new.ups[-1].stride,
                                  padding=new.ups[-1].padding)
    torch.nn.init.zeros_(up.weight), torch.nn.init.zeros_(up.bias)
    with torch.enable_grad():
        opt = torch.optim.Adam(up.parameters(), lr=1e-3)
        for it in range(600):
            loss = sum(F.mse_loss(up(r["s_ups_in"]), r["t_ups"][:, idx]) for r in fit_rows) / len(fit_rows)
            opt.zero_grad()
            loss.backward()
            opt.step()
    r2 = lambda rows: float(1 - sum(F.mse_loss(up(r["s_ups_in"]), r["t_ups"][:, idx]) for r in rows)
                            / sum(r["t_ups"][:, idx].var() for r in rows))
    ups_r2 = dict(fit=r2(fit_rows), held=r2(held_rows))
    print(f"ups.1 fit R2 {ups_r2}", flush=True)
    new.ups[-1].weight[:, base:] = up.weight
    new.ups[-1].bias[base:] = up.bias

    # conv_post new columns: choose on held-out batches
    def post_inputs(rows):
        out = []
        for r, (asr, f0, n, s, har) in zip(rows, fit if rows is fit_rows else held):
            _, got = capture(new, ["conv_post"], lambda: new.logits(asr, f0, n, s, har))
            out.append(got["conv_post"][0])
        return out

    new.conv_post.weight[:, base:] = 0
    x_fit, x_held = post_inputs(fit_rows), post_inputs(held_rows)
    same = max(float((F.conv1d(x, new.conv_post.weight, new.conv_post.bias, padding=3) - r["ls"]).abs().max())
               for x, r in zip(x_held, held_rows))
    assert same < 1e-4, f"zero new columns should reproduce v2 exactly, max diff {same}"

    teacher_cols = tt["conv_post.weight"][:, idx]
    # ridge fit of the new columns to the residual, on unfolded inputs [B*T, extra*7]
    unfold = lambda x: F.unfold(F.pad(x[:, base:], (3, 3))[:, :, None], (1, 7)).transpose(1, 2).reshape(-1, extra * 7)
    A = torch.cat([unfold(x) for x in x_fit])
    Y = torch.cat([(r["lt"] - r["ls"]).transpose(1, 2).reshape(-1, 22) for r in fit_rows])
    lam = 1e-2 * A.pow(2).mean() * A.shape[0]
    sol = torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1]), A.T @ Y)  # [extra*7, 22]
    ridge_cols = sol.T.reshape(22, extra, 7)

    candidates = {"zero": torch.zeros_like(teacher_cols), "teacher_x0.25": 0.25 * teacher_cols,
                  "teacher_x0.5": 0.5 * teacher_cols, "teacher_x1": teacher_cols, "ridge": ridge_cols}
    scores = {}
    for name, cols in candidates.items():
        wt = new.conv_post.weight.clone()
        wt[:, base:] = cols
        scores[name] = float(np.mean([logmag_l1(F.conv1d(x, wt, new.conv_post.bias, padding=3), r["lt"])
                                      for x, r in zip(x_held, held_rows)]))
    choice = min(scores, key=scores.get)
    new.conv_post.weight[:, base:] = candidates[choice]
    print(f"held-out log-mag L1 {json.dumps({k: round(v, 5) for k, v in scores.items()})} -> {choice}", flush=True)

    graft = dict(source=str(args.init), base=base, last=args.last, max_corr=args.max_corr, picked=picked,
                 skipped_stronger=skipped, strength=strength[idx].tolist(),
                 max_corr_to_v2=corr_old[idx].max(1).values.tolist(), ups_r2=ups_r2, conv_post_scores=scores,
                 conv_post_choice=choice, fit_batches=len(fit), held_batches=len(held))
    keep = {k: state[k] for k in ("pitch", "pitch_config", "disc", "optimizer_d") if k in state}
    torch.save({**keep, "vocoder": new.state_dict(), "vocoder_config": new.config, "step": state["step"],
                "graft": graft}, run / "init.pt")
    (run / "graft.json").write_text(json.dumps(graft, indent=2))
    print(f"wrote {run / 'init.pt'} step {state['step']} config {new.config}", flush=True)


if __name__ == "__main__":
    main()
