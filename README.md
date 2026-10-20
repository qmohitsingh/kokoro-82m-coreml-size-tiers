# Kokoro size lab (S017)

Offline [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) Core ML model bundles at progressively smaller size tiers, checked on a physical iPhone 12. This repository holds the harness, build specs, scripts, frozen inputs, raw results, and experiment log that reproduce **all five Hugging Face bundles** (500, 200, 100, 50 and 25 MB). **S017, the 25 MB tier (24,267,607 bytes of compiled Core ML models), is the recommended default and the smallest.** The models themselves are in the Hugging Face model repository [`qmohits/kokoro-82m-coreml-size-tiers`](https://huggingface.co/qmohits/kokoro-82m-coreml-size-tiers) (see "Get the models").

This is an iPhone 12 / Core ML engineering case study: one device, one voice (`af_heart`), one main listener (a second listener ran the 25mb four-length test). It does not claim broad perceptual equivalence to Kokoro. The iOS app is a validation harness, not a product, and there is **no English text-to-phoneme frontend**: input is Kokoro phoneme token ids.

## Tiers

Each tier is runnable here: a project spec, the shared Swift harness, and a `Resources-*` folder whose symlinks point into `artifacts/hf/<tier>/` (filled by `scripts/download_models.py --tier <tier>`). Only tiers that passed speed and sound are included; failed and unscored experiments are not released as models (they remain described in the log).

| Tier folder | Experiment | Compiled models (bytes) | Shapes | Packages | Median RTF 3/7/15/30 s | Sound |
| --- | --- | ---: | --- | ---: | --- | --- |
| `500mb/` | S002 | 327,680,270 | fixed | 5 | 0.170 / 0.198 / 0.207 / 0.215 | 4/4 ties |
| `200mb/` | S007 | 168,492,559 | fixed | 5 | 0.185 / 0.211 / 0.223 / 0.269 | 4/4 ties |
| `100mb/` | S011 | 96,203,053 | fixed | 5 | 0.171 / 0.196 / 0.213 / 0.226 | 4/4 ties |
| `50mb/` | S014 mixed8 | 42,582,199 | flexible | 4 | 0.173 / 0.181 / 0.189 / 0.197 | 12/12 ties |
| `25mb/` (recommended) | S017 | 24,267,607 | flexible | 3 | 0.060 / 0.072 / 0.096 / 0.109 | 11/16 ties, 4 S017 wins, 1 reference win (owner call) |

The 25mb sound count covers the counted hidden-sentence session (8 unseen sentences, 8 ties) and the four-length test (3 ties, 1 S017 win), 12 pairs from listener-1 with no reference win, so the written gate passes for that listener. An earlier 8-pair hidden session without the play-to-end gate is in the log and is not counted. A second listener (listener-2) ran the four-length test only: 3 S017 preferences and 1 reference preference (15 s). Under the strict written rule that is one reference preference; the owner judges it acceptable because the smaller model was preferred in 3 of 4 pairs. listener-1 is the project author; the test was blind but not independent, and the author judged listener-2's single reference preference acceptable. 500mb to 100mb passed the four-length test, 50mb passed four lengths plus 8 hidden sentences.

| Tier | Project spec | Swift sources | Resources (links into `artifacts/hf/<tier>/`) |
| --- | --- | --- | --- |
| `500mb` S002 | `project-s002.yml` | `KokoroSize/` (flag `S002`) | `Resources-S002/` |
| `200mb` S007 | `project-s007.yml` | `KokoroSize/` (flag `S007`) | `Resources-S007/` |
| `100mb` S011 | `project-s011.yml` | `KokoroSize/` (flag `S011`) | `Resources-S011/` |
| `50mb` S014 | `project-s014.yml` | `KokoroSize/` (flag `S014`) | `Resources-S014/` |
| `25mb` S017 | `project-s017.yml` | `KokoroSize-S017/` | `Resources-S017/` |

**Fixed shapes:** the 500mb, 200mb and 100mb tiers are fixed-shape multifunction packages and support only the five frozen benchmark shapes (token functions `t56, t110, t237, t260, t280`; frame functions `f146, f275, f567, f614, f633`). Flexible token and frame lengths start at the 50mb tier.

**Compute units** (read from each tier's `CoreMLChain.swift`): every package loads with `.all`, except the 50mb generator (`kokoro_generator_mixed8_flex`, `.cpuAndGPU`) and the 25mb vocoder (`kokoro_s015_vocoder_int8`, `.cpuOnly`). **Model interfaces** (names, dtypes, shapes, function names) are listed per package in `MANIFEST.json` and in each tier's `manifest.json` on Hugging Face; a summary follows.

No ONNX Runtime and no ONNX model are part of any harness: the Swift sources contain no ONNX code and the Resources folders link only Core ML packages, frozen WAV references and `bench.json`.

### 500mb (S002, Baseline (at most 500 MB))
Compiled models total **327,680,270** bytes. Kokoro-82M v1.0 split into five stages, fp32, with the five shapes of each stage sharing one weight file (coremltools save_multifunction).
**Fixed shapes only.** This tier is built from fixed-shape multifunction packages. It supports ONLY the five frozen benchmark shapes (token functions t56, t110, t237, t260, t280 and frame functions f146, f275, f567, f614, f633). Arbitrary text lengths are not supported. Flexible token and frame lengths start at the 50mb tier (S014).

| Package | Compiled bytes | Compute units | Interface |
| --- | ---: | --- | --- |
| `kokoro_text_fp32.mlpackage` | 25,469,752 | `.all` | functions t56, t110, t237, t260, t280; e.g. t56: input_ids int32[1, 56] -> encoded float32[1, 512, 56] |
| `kokoro_prosody_fp32.mlpackage` | 53,708,437 | `.all` | functions t56, t110, t237, t260, t280; e.g. t56: input_ids int32[1, 56], encoded float32[1, 512, 56], style float32[1, 256] -> duration_hidden float32[1, 56, 640], duration_logits float32[1, 56, 50], text_features float32[1, 512, 56] |
| `kokoro_pitch_fp32.mlpackage` | 33,959,547 | `.all` | functions f146, f275, f567, f614, f633; e.g. f146: aligned float32[1, 640, 146], style float32[1, 128] -> pitch float32[1, 292], energy float32[1, 292] |
| `kokoro_decoder_fp32.mlpackage` | 134,687,955 | `.all` | functions f146, f275, f567, f614, f633; e.g. f146: text float32[1, 512, 146], pitch float32[1, 292], energy float32[1, 292], style float32[1, 128] -> features float32[1, 512, 292] |
| `kokoro_generator_fp32.mlpackage` | 79,854,579 | `.all` | functions f146, f275, f567, f614, f633; e.g. f146: features float32[1, 512, 292], style float32[1, 128], spectrum float32[1, 22, 17521] -> magnitude float32[1, 11, 17521], phase float32[1, 11, 17521] |

Median RTF (iPhone 12): 3s 0.170, 7s 0.198, 15s 0.207, 30s 0.215 (gate 0.30; source `benchmark/results/size-shared-speed.json`).
Sound: 4 ties of 4 blind pairs, 0 reference preferred (`benchmark/results/size-shared-listen.json`).
Stability: 3607 s, 323 loops, finished (`benchmark/results/size-shared-stability.json`).
Voice: no voice file; style rows come from the frozen `benchmark/inputs/bench.json`.

### 200mb (S007, Stretch (at most 200 MB))
Compiled models total **168,492,559** bytes. S002 stack with the decoder and generator weights quantized to int8 (per output channel, weight-only). Text, prosody and pitch stay fp32.
**Fixed shapes only.** This tier is built from fixed-shape multifunction packages. It supports ONLY the five frozen benchmark shapes (token functions t56, t110, t237, t260, t280 and frame functions f146, f275, f567, f614, f633). Arbitrary text lengths are not supported. Flexible token and frame lengths start at the 50mb tier (S014).

| Package | Compiled bytes | Compute units | Interface |
| --- | ---: | --- | --- |
| `kokoro_text_fp32.mlpackage` | 25,469,752 | `.all` | functions t56, t110, t237, t260, t280; e.g. t56: input_ids int32[1, 56] -> encoded float32[1, 512, 56] |
| `kokoro_prosody_fp32.mlpackage` | 53,708,437 | `.all` | functions t56, t110, t237, t260, t280; e.g. t56: input_ids int32[1, 56], encoded float32[1, 512, 56], style float32[1, 256] -> duration_hidden float32[1, 56, 640], duration_logits float32[1, 56, 50], text_features float32[1, 512, 56] |
| `kokoro_pitch_fp32.mlpackage` | 33,959,547 | `.all` | functions f146, f275, f567, f614, f633; e.g. f146: aligned float32[1, 640, 146], style float32[1, 128] -> pitch float32[1, 292], energy float32[1, 292] |
| `kokoro_decoder_int8.mlpackage` | 34,151,789 | `.all` | functions f146, f275, f567, f614, f633; e.g. f146: text float32[1, 512, 146], pitch float32[1, 292], energy float32[1, 292], style float32[1, 128] -> features float32[1, 512, 292] |
| `kokoro_generator_int8.mlpackage` | 21,203,034 | `.all` | functions f146, f275, f567, f614, f633; e.g. f146: features float32[1, 512, 292], style float32[1, 128], spectrum float32[1, 22, 17521] -> magnitude float32[1, 11, 17521], phase float32[1, 11, 17521] |

Median RTF (iPhone 12): 3s 0.185, 7s 0.211, 15s 0.223, 30s 0.269 (gate 0.30; source `benchmark/results/size-int8-dg-speed.json`).
Sound: 4 ties of 4 blind pairs, 0 reference preferred (`benchmark/results/size-int8-dg-listen.json`).
Stability: 3625 s, 146 loops, finished (`benchmark/results/size-int8-dg-stability.json`).
Voice: no voice file; style rows come from the frozen `benchmark/inputs/bench.json`.

### 100mb (S011, Aggressive (at most 100 MB))
Compiled models total **96,203,053** bytes. Int8 text embeddings, int4 prosody text branch, int8 pitch, int4 decoder and generator. Activations and I/O stay fp32.
**Fixed shapes only.** This tier is built from fixed-shape multifunction packages. It supports ONLY the five frozen benchmark shapes (token functions t56, t110, t237, t260, t280 and frame functions f146, f275, f567, f614, f633). Arbitrary text lengths are not supported. Flexible token and frame lengths start at the 50mb tier (S014).

| Package | Compiled bytes | Compute units | Interface |
| --- | ---: | --- | --- |
| `kokoro_text-embedding-int8.mlpackage` | 24,392,490 | `.all` | functions t56, t110, t237, t260, t280; e.g. t56: input_ids int32[1, 56] -> encoded float32[1, 512, 56] |
| `kokoro_prosody-text-int4.mlpackage` | 34,156,234 | `.all` | functions t56, t110, t237, t260, t280; e.g. t56: input_ids int32[1, 56], encoded float32[1, 512, 56], style float32[1, 256] -> duration_hidden float32[1, 56, 640], duration_logits float32[1, 56, 50], text_features float32[1, 512, 56] |
| `kokoro_pitch_int8.mlpackage` | 8,892,606 | `.all` | functions f146, f275, f567, f614, f633; e.g. f146: aligned float32[1, 640, 146], style float32[1, 128] -> pitch float32[1, 292], energy float32[1, 292] |
| `kokoro_decoder-all-int4.mlpackage` | 17,369,455 | `.all` | functions f146, f275, f567, f614, f633; e.g. f146: text float32[1, 512, 146], pitch float32[1, 292], energy float32[1, 292], style float32[1, 128] -> features float32[1, 512, 292] |
| `kokoro_generator-all-int4.mlpackage` | 11,392,268 | `.all` | functions f146, f275, f567, f614, f633; e.g. f146: features float32[1, 512, 292], style float32[1, 128], spectrum float32[1, 22, 17521] -> magnitude float32[1, 11, 17521], phase float32[1, 11, 17521] |

Median RTF (iPhone 12): 3s 0.171, 7s 0.196, 15s 0.213, 30s 0.226 (gate 0.30; source `benchmark/results/size-s011-speed.json`).
Sound: 4 ties of 4 blind pairs, 0 reference preferred (`benchmark/results/size-s011-listen.json`). Listener note: some clips sounded slightly lower in volume, not enough to prefer either side (measured RMS about 0.5 dB below the reference). The later 120-sentence test of the shared int4 decoder/generator found a metallic edge (see S014 pilot in the log); this tier was only scored with the four-length test.
Stability: not run: the 60-minute loop was made optional on 2026-10-16 and is not a gate.
Voice: no voice file; style rows come from the frozen `benchmark/inputs/bench.json`.

### 50mb (S014 mixed8, Research (at most 50 MB))
Compiled models total **42,582,199** bytes. S014 width-256 student frontend, Kokoro int8 pitch, int4 decoder, mixed int4/int8 generator, all flexible length.
Flexible token length (1 to 512 tokens) and frame length: the packages accept any length in range, so arbitrary sentences work once phoneme ids are supplied.

| Package | Compiled bytes | Compute units | Interface |
| --- | ---: | --- | --- |
| `kokoro_student_w256.mlpackage` | 6,076,253 | `.all` | input_ids int32[1, 1] range [[1, 1], [1, 512]], style float32[1, 256] -> duration_hidden float16[dynamic], duration_logits float16[dynamic], text_features float16[dynamic] |
| `kokoro_pitch_int8_flex.mlpackage` | 8,622,252 | `.all` | aligned float32[1, 640, 1] range [[1, 1], [640, 640], [1, 2048]], style float32[1, 128] -> pitch float32[dynamic], energy float32[dynamic] |
| `kokoro_decoder_int4_flex.mlpackage` | 17,124,260 | `.all` | text float32[1, 512, 1] range [[1, 1], [512, 512], [1, 2048]], pitch float32[1, 2] range [[1, 1], [2, 4096]], energy float32[1, 2] range [[1, 1], [2, 4096]], style float32[1, 128] -> features float32[dynamic] |
| `kokoro_generator_mixed8_flex.mlpackage` | 10,759,434 | `.cpuAndGPU` | features float32[1, 512, 2] range [[1, 1], [512, 512], [2, 4096]], style float32[1, 128], spectrum float32[1, 22, 121] range [[1, 1], [22, 22], [121, 245761]] -> magnitude float32[dynamic], phase float32[dynamic] |

Median RTF (iPhone 12): 3s 0.173, 7s 0.181, 15s 0.189, 30s 0.197 (gate 0.30; source `benchmark/results/size-s014-m8-speed.json`).
Sound: 12 ties of 12 blind pairs, 0 reference preferred (`benchmark/results/size-s014-m8-listen.json`). The 12 ties were the 4 four-length pairs plus 8 hidden-sentence pairs.
Stability: not run: the 60-minute loop was made optional on 2026-10-16 and is not a gate.
Voice: no voice file; style rows come from the frozen `benchmark/inputs/bench.json`.

### 25mb (S017, Ultra (at most 25 MB))
**Recommended default tier.**
Compiled models total **24,267,607** bytes. S014 width-256 student, Kokoro int8 flexible pitch, pruned and distilled student vocoder (int8). The vocoder runs CPU-only.
Flexible token length (1 to 512 tokens) and frame length: the packages accept any length in range, so arbitrary sentences work once phoneme ids are supplied.

| Package | Compiled bytes | Compute units | Interface |
| --- | ---: | --- | --- |
| `kokoro_student_w256.mlpackage` | 6,076,253 | `.all` | input_ids int32[1, 1] range [[1, 1], [1, 512]], style float32[1, 256] -> duration_hidden float16[dynamic], duration_logits float16[dynamic], text_features float16[dynamic] |
| `kokoro_pitch_int8_flex.mlpackage` | 8,622,252 | `.all` | aligned float32[1, 640, 1] range [[1, 1], [640, 640], [1, 2048]], style float32[1, 128] -> pitch float32[dynamic], energy float32[dynamic] |
| `kokoro_s015_vocoder_int8.mlpackage` | 9,569,102 | `.cpuOnly` | text float32[1, 512, 1] range [[1, 1], [512, 512], [1, 2048]], pitch float32[1, 2] range [[1, 1], [2, 4096]], energy float32[1, 2] range [[1, 1], [2, 4096]], style float32[1, 128], spectrum float32[1, 22, 121] range [[1, 1], [22, 22], [121, 245761]] -> magnitude float32[dynamic], phase float32[dynamic] |

Median RTF (iPhone 12): 3s 0.060, 7s 0.072, 15s 0.096, 30s 0.109 (gate 0.30; source `benchmark/results/size-s017-speed.json`).
Sound: 11 ties of 16 blind pairs, 4 preferred the tier's output, 1 reference preferred (`benchmark/results/size-s017-listen.json, benchmark/results/size-s017-listen-four.json`). listener-1: 12 pairs, 11 ties, 1 S017 preference, 0 reference preferences, so the written rule passes. listener-2 (four lengths): 3 S017 preferences and 1 reference preference (15 s). Under the strict written rule that session has one reference preference; the owner judges this acceptable because the smaller model was preferred in 3 of 4 pairs. listener-1 is the project author; the test was blind but not independent, and the author judged listener-2's single reference preference acceptable.
Stability: not run: the 60-minute loop is not a gate.
Voice: `af_heart_f32.bin` (522,240 bytes), models plus voice 24,789,847.

## Result (S017, the recommended tier)

Sizes are reported separately:

| Item | Bytes |
| --- | ---: |
| Compiled Core ML models (student 6,076,253, pitch 8,622,252, vocoder 9,569,102) | **24,267,607** |
| `af_heart` fp32 voice table (`af_heart_f32.bin`) | 522,240 |
| Models plus voice table | 24,789,847 |
| Harness `.app`, unsigned Release (recorded, not graded) | 24,976,568 |
| Harness `.app`, Listen configuration (not graded) | 25,110,928 unsigned, 25,149,782 signed |


Gates (see [`docs/GOAL.md`](docs/GOAL.md)):

| Gate | Status |
| --- | --- |
| Speed: warm median RTF 0.30 or lower at 3, 7, 15, 30 s | Pass: 0.060, 0.072, 0.096, 0.109 (iPhone 12, Release, 2 warm-ups, median of 10, charging, thermal state serious) |
| Sound: one listener, blind, against reference Kokoro `af_heart`; passes unless the reference is preferred | Short hidden-sentence session (8 unseen sentences, 2 to 4 s clips, randomized sides, play-to-end gate): 8 ties, 0 reference wins. Documented four-length blind test (3, 7, 15, 30 s): 3 ties, 1 S017 preference (30 s), 0 reference wins. Both meet the rule for listener-1. Listener-2, four lengths only: 3 S017 wins, 1 reference win (15 s). **Accepted by owner judgement** (smaller model preferred in 3 of 4). listener-1 is the project author; the test was blind but not independent, and the author judged listener-2's single reference preference acceptable. |
| Size: compiled models | 24,267,607 bytes (25 MB Ultra tier met) |

| Passage | Audio (s) | Median RTF | p90 RTF |
| --- | ---: | ---: | ---: |
| 3s | 3.575 | 0.060 | 0.062 |
| 7s | 6.800 | 0.072 | 0.074 |
| 15s | 14.175 | 0.096 | 0.101 |
| 30s | 31.250 | 0.109 | 0.111 |

Objective check (recorded, not a gate): mean log-mel distance against fp32 Kokoro over 15 items for the full shipped chain, all three models in Core ML, CPU only on a Mac: **1.787 dB against the 1.86 dB threshold (PASS)** (`benchmark/results/size-s017-eval.json`; highest items val8 2.168, 30s-0 1.987, 15s 1.918). The earlier 1.747 dB figure used a PyTorch student and is superseded. On the phone the student and pitch run on `.all`, so this is not an on-device number. Mac Core ML frame counts equal the phone's on the benchmark cases and on all 120 listening clips.

In the speed result JSON the field `sampleCountMatched` was later renamed `outputShapeValid` in the harness (non-empty audio, a whole number of 600-sample frames, same length across runs). `benchmark/results/size-s017-speed.json` keeps the old key.

## The chain

```
token ids + af_heart style row
  -> kokoro_student_w256 (.all)        S014 width-256 student: text body and durations
  -> Swift durations + alignment
  -> kokoro_pitch_int8_flex (.all)     Kokoro pitch/energy, int8, flexible length
  -> Swift SourceDSP
  -> kokoro_s015_vocoder_int8 (CPU)    v4-prune student vocoder, int8
  -> Swift inverse STFT -> 24 kHz audio
```

The vocoder must run CPU-only on the tested iPhone 12: `.cpuAndGPU` got the app killed by iOS at 567 frames and above.

## Repository layout

- `scripts/`: conversion (`convert_*`, `share_fixed_shapes.py` for the fixed-shape tiers), compression (`compress_*`, for the 200mb and 100mb tiers), distillation, pruning, evaluation, and the S017 tools (`download_models.py`, `s017_export_voice.py`, `s017_bench_inputs.py`, `s017_eval_full_chain.py`, `s017_four_length_prep.py`). The compression scripts read the separate fixed-shape fp32 packages from the earlier conversion project [`kokoro-82m-coreml`](https://github.com/qmohitsingh/kokoro-82m-coreml) (pass the folder with `--fp32-source`; default `artifacts/fp32-source/`); they were not re-run for this release. The closed S016 attempt is not included.
- `benchmark/inputs/`: frozen token ids, voice rows, and reference audio (`bench.json`, `bench-s017.json`).
- `benchmark/ios/`: the shared `KokoroSize/` harness (tiers 500mb to 50mb), `KokoroSize-S017/`, the five `project-s*.yml` specs, `Config-S*.xcconfig`, and `Resources-S*` link folders.
- `benchmark/results/`: raw result JSON (listener names replaced with `listener-N` labels, which merge the owner's sessions; session ids rewritten to `pair-NNN`).
- `docs/`: `GOAL.md`, `EXPERIMENT_LOG.md`, `MODEL_CARD.md` (pointer).
- `MANIFEST.json`: per-tier packages, compiled sizes, interfaces, compute units, metrics, and the size and SHA-256 of every model file, the voice table, and the bench files.
- `THIRD_PARTY_NOTICES.md`: upstream attribution, provenance, and licences.

Models, checkpoints, corpora, build products, and virtual environments are not in Git. The S002/S007 harness sources were patched in this release only by `#if` branches (adding the `S002` flag for the fp32 decoder and generator, and removing the unscored S012 branch). The trained student and vocoder checkpoints are published in the Hugging Face model repository under `checkpoints/`.

## Requirements

Pinned versions used for the result: macOS with Xcode 26.1.1 (17B100); xcodegen 2.46.0; Python 3.12.13; coremltools 9.0; torch 2.7.0; kokoro 0.9.4 (misaki 0.9.4); transformers 4.46.3; numpy 2.5.4; soundfile 0.14.0; huggingface_hub 0.36.2 (all in `requirements-convert.txt`). A physical iPhone running iOS 18 or newer is needed for speed and listening tests.

```sh
python3.12 -m venv .venv-convert
.venv-convert/bin/pip install -r requirements-convert.txt
```

## Get the models

```sh
.venv-convert/bin/python scripts/download_models.py qmohits/kokoro-82m-coreml-size-tiers                    # default: --tier 25mb
.venv-convert/bin/python scripts/download_models.py qmohits/kokoro-82m-coreml-size-tiers --tier 100mb       # one tier: 500mb 200mb 100mb 50mb 25mb
.venv-convert/bin/python scripts/download_models.py qmohits/kokoro-82m-coreml-size-tiers --tier all         # optional: --revision <commit>
```

This downloads into `artifacts/hf/<tier>/` (ignored by Git) and verifies every file's size and SHA-256 against `MANIFEST.json`. The `benchmark/ios/Resources-*` folders are made of relative symlinks into those directories, so the model links dangle until the matching tier has been downloaded.

## Build a tier's harness (unsigned)

Example for the recommended tier; substitute `s002`/`S002`, `s007`/`S007`, `s011`/`S011` or `s014`/`S014` and the tier folder for the others.

```sh
cd benchmark/ios
xcodegen generate --spec project-s017.yml
xcodebuild -project KokoroSizeS017.xcodeproj -scheme KokoroSize -configuration Release \
  -destination 'generic/platform=iOS' -derivedDataPath ../../build/s017 CODE_SIGNING_ALLOWED=NO build
```

The `.app` is `build/s017/Build/Products/Release-iphoneos/KokoroSize.app`. Graded size: sum the logical bytes of its `.mlmodelc` folders (`python scripts/logical_size.py <folder>`); for 25mb the total must be 24,267,607. Report the voice table (`af_heart.bin`, 522,240, 25mb only) and the whole `.app` separately.

No signing configuration is needed for an unsigned build. For a device, copy `benchmark/ios/Signing.example.xcconfig` to `benchmark/ios/Signing.local.xcconfig` (git-ignored); every tier's `Config-<tag>.xcconfig` includes it when present.

## Install and validate on an iPhone 12

1. Create `benchmark/ios/Signing.local.xcconfig` from the example (ignored by Git) and set your team and a bundle id your team can sign (the default is `com.kokorobench.KokoroBench`). Regenerate the project and build with `-allowProvisioningUpdates` and no `CODE_SIGNING_ALLOWED=NO`, or sign in Xcode.
2. Install and run (find the device id with `xcrun devicectl list devices`):

```sh
xcrun devicectl device install app --device <device-id> <path-to>/KokoroSize.app
xcrun devicectl device process launch --device <device-id> --console <bundle-id> -- -coreml-speed
```

3. The speed test (2 warm-ups, median of 10 at 3, 7, 15, 30 s) writes `coreml-speed.json` to the app's Documents (visible in the Files app). Pull it with `xcrun devicectl device copy from`. Keep the phone charging and note the thermal state in the result.
4. `-- -hash` writes the SHA-256 of the 3 s waveform for comparing configurations.
5. Listening: build the `Listen` configuration. Clips go in the app's Documents and the app has a "Short hidden | Four lengths" picker. See `scripts/s014_listen_prep.py`, `scripts/s017_four_length_prep.py`, and the log. The four-length reference clips are `benchmark/inputs/*.wav`.

## Reproduce the voice table and bench inputs (25mb)

Download the upstream voice (Hugging Face `hexgrad/Kokoro-82M`, revision `f3ff3571791e39611d31c381e3a41a3af07b4987`, `voices/af_heart.pt`, SHA-256 `0ab5709b8ffab19bfd849cd11d98f75b60af7733253ad0d67b12382a102cb4ff`) to `models/pytorch/voices/af_heart.pt`, and `kokoro-v1_0.pth` plus `config.json` to `models/pytorch/` (checkpoint SHA-256 `496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4`). Then:

```sh
.venv-convert/bin/python scripts/s017_export_voice.py     # artifacts/s017/af_heart_f32.bin  (522,240 bytes)
.venv-convert/bin/python scripts/s017_bench_inputs.py     # artifacts/s017/bench-s017.json
```

Both outputs match the hashes in `MANIFEST.json` and equal the files in the Hugging Face repository and in the app (`af_heart_f32.bin` is the app's `af_heart.bin`).

## Rebuild the models and rerun the checks

Only the 25mb lineage was audited for this release; the older tiers' scripts are included for reference. The full lineage is in `docs/EXPERIMENT_LOG.md` (sections S014, S015, S017). In order: `s014_corpus.py` and `s014_templates.py` (training text), `s014_cache_teacher.py`, `s014_train.py`, `s014_convert_student.py` (student), `s014_convert_downstream.py` (flexible int8 pitch), `s015_cache_pitch.py`, `s015_train.py`, `s015_widen.py`, `s015_prune.py`, `s015_train_gan.py` (vocoder), `s015_convert.py` (int8 vocoder). Retraining needs a GPU for the vocoder and will not be byte-identical. The trained checkpoints are published in the model repository's `checkpoints/` folder, so the conversion scripts (`s014_convert_student.py`, `s015_convert.py`) can be re-run without retraining.

Objective check on a Mac:

```sh
.venv-convert/bin/python scripts/s017_eval_full_chain.py     # writes benchmark/results/size-s017-eval.json
```

## Licence

[Apache-2.0](LICENSE). Kokoro code and weights are Apache-2.0. See `THIRD_PARTY_NOTICES.md`. Do not redistribute espeak-ng output corpora or other third-party data without checking its licence.

## Future work

Target about **12.1 MB of compiled models** with the same speed and sound gates. Also: more voices and devices, and an English phoneme frontend.

## Citation

The study behind these tiers is described in the paper *Compact On-Device Speech Synthesis with Open-Weight Neural TTS: A Case Study of Kokoro-82M* (preprint; arXiv ID to be added on submission). GitHub also renders `CITATION.cff` as a "Cite this repository" button.

```bibtex
@misc{singh2026kokorocoreml,
  title  = {Compact On-Device Speech Synthesis with Open-Weight Neural TTS: A Case Study of Kokoro-82M},
  author = {Singh, Mohit},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/qmohitsingh/kokoro-82m-coreml-size-tiers}
}
```
