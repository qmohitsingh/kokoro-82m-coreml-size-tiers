# Experiment log

> **Release note.** This log is the working record of the project, scrubbed of personal names, accounts, hosts, and local paths for publication ("the owner" is the project owner; listeners appear as `listener-N`; GPU hosts as `<gpu-vm>`). Entries are historical and keep the wording used when they were written, including the word "accepted" for S017. The current status, with the sizes kept separate, is in `README.md`: compiled models 24,267,610 bytes; `af_heart` table 522,240 bytes; harness `.app` 24,976,571 bytes (unsigned Release). Listener-1 ran the counted short hidden-sentence session (8 ties, 0 reference wins; the earlier session 1 without the play-to-end gate is not counted) and the four-length blind test (3 ties, 1 S017 win, 0 reference wins), so the complete documented sound gate passes for that listener. A second listener (listener-2) ran the four-length test only: 3 S017 preferences and 1 reference preference (15 s). Under the strict written rule that is one reference preference; the author judges it acceptable because the smaller model was preferred in 3 of 4 pairs. Listener-1 is the project author; the test was blind but not independent, and the author judged listener-2's single reference preference acceptable. Where an older entry says the four-length test was "prepared, not run", the later "S017 four-length blind test" entry supersedes it. The full-chain objective result is in `benchmark/results/size-s017-eval.json`: 1.787 dB (PASS) and supersedes the earlier 1.747 dB figure, which came from a run that used the PyTorch student. The last entry, S018, is a summary written for publication; the S017 entries above are unchanged. The 25mb and 50mb tiers now carry the S018 student (compiled models 24,267,610 bytes for 25mb).


Sizes are logical file bytes (`os.stat`), not APFS allocated blocks. The Release `.app` is an unsigned arm64 iOS build (`generic/platform=iOS`, `CODE_SIGNING_ALLOWED=NO`). Early sections were measured before the iPhone was available. From S003 on, speed, listening, and stability were run on the iPhone 12.

Graded speed, listen, and stability JSON copied from the earlier conversion project [`kokoro-82m-coreml`](https://github.com/qmohitsingh/kokoro-82m-coreml) are not results for this size build. Size-lab scores use a `size-` prefix.

## Current result

The accepted result is S017: 24,267,607 compiled-model bytes, or 24,789,847 bytes including the required fp32 `af_heart` table (522,240). It passes speed (median RTF 0.060-0.109) and the one-listener sound gate in `GOAL.md` (counted short session: 8 ties, 0 reference wins; the earlier session without the play-to-end gate is not counted; four-length test: 3 ties, 1 S017 win, 0 reference wins) and is under the 25 MB Ultra tier. The previous result was S014 mixed8 at 42,582,199 compiled-model bytes (12 ties, 0 reference wins). The 60-minute loop was not run. The earlier int4-generator app failed the ear test and is not the result.

| Build | Unsigned `.app` | Speed | Sound |
| --- | ---: | --- | --- |
| S007, shared fp32 front end, int8 decoder and generator | 171.7 MB | pass | pass, 4 ties |
| S011, earlier 100 MB result | 99,489,355 | pass | pass, 4 ties |
| S014 mixed8, student plus flexible downstream | 46,107,717 | pass | pass, 12 ties |
| S017, S014 frontend + int8 pitch + v4-prune vocoder | 24,976,568 | pass | pass, 8 ties |

S017 is graded on 24,267,607 compiled-model bytes (frontend 6,076,253, pitch
8,622,252, vocoder 9,569,102), and 24,789,847 including `af_heart`. The `.app`
sizes in the table are the validation harness, recorded separately and not graded.
Hugging Face and GitHub publication stay deferred until licensing and model
provenance are reviewed.

## S001 — Core ML-only baseline

Separate fp32 package per shape. ONNX model and ONNX Runtime removed. Text and duration stay fp32.

| Piece | Bytes |
| --- | ---: |
| Release `KokoroSize.app` | 1,628,598,173 |
| Five decoder bodies | 672,307,537 |
| Five generators | 394,614,869 |
| Five prosody models | 267,700,377 |
| Five pitch models | 168,550,340 |
| Five text models | 121,957,510 |
| Reference wavs, JSON, binary, plist | 3,467,540 |

No `.onnx` file and no ONNX Runtime linkage. 1,628.6 MB. Above 500 MB.

## S002 — Multifunction weight sharing

`coremltools` 9.0 `save_multifunction`, then `coremlcompiler`. Each stage is one package with functions `t56`, `t110`, `t237`, `t260`, `t280` or `f146`, `f275`, `f567`, `f614`, `f633`.

Compiled `weight.bin` in the app matches the compiler output byte for byte. One weight file per stage. Compilation did not duplicate the weights.

| Stage | Five `weight.bin` bytes | Shared `weight.bin` bytes | Ratio | Compiled model in the app |
| --- | ---: | ---: | ---: | ---: |
| text | 121,574,336 | 25,091,008 | 0.2064 | 25,469,752 |
| prosody | 267,487,040 | 53,497,408 | 0.2000 | 53,708,437 |
| pitch | 168,234,240 | 33,646,848 | 0.2000 | 33,959,547 |
| decoder | 672,020,800 | 134,404,160 | 0.2000 | 134,687,955 |
| generator | 393,446,720 | 78,689,344 | 0.2000 | 79,854,579 |

Text is the only stage above a fifth. Its position tables differ by length, so those constants stay separate. The other four weight files are one exact copy.

Random CPU inputs, every function, every output: max absolute error `0` against the separate package. Frozen 3s token ids through the shipped text model `t56`: max absolute error `0`.

| Piece | Bytes |
| --- | ---: |
| Release `KokoroSize.app` | 330,883,802 |
| Five compiled models | 327,680,270 |
| Reference wavs, JSON, binary, plist | 3,203,532 |

330.9 MB. Under the 500 MB tier. fp16 and palettization were not tried.

## S003 — iPhone 12 speed, shared fp32

Installed on the author's iPhone 12 (`iPhone13,2`, iOS 26.6.2). The CLI has no Apple account, so the app was signed with the existing `com.kokorobench.KokoroBench` profile. That replaced the graded bench app on the phone. The product name is still KokoroSize.

Protocol: discard 2 warmups, median of 10, model load outside the clock. Raw file: `benchmark/results/size-shared-speed.json`.

| Case | Median RTF | Sample count | Thermal |
| --- | ---: | --- | --- |
| 3s | 0.170 | matched | nominal |
| 7s | 0.198 | matched | nominal |
| 15s | 0.207 | matched | nominal |
| 30s | 0.215 | matched | nominal |

Every length is under 0.30.

## S004 — iPhone 12 listening, shared fp32

Four blind pairs, same sentences and `af_heart`. Raw file: `benchmark/results/size-shared-listen.json`.

| Case | Mark | Winner |
| --- | --- | --- |
| 3s | tie | neither |
| 7s | tie | neither |
| 15s | tie | neither |
| 30s | tie | neither |

Reference wins 0. The sound gate passes.

## S005 — iPhone 12 stability, shared fp32

The first tap was interrupted by a rebuild and is not the score. The restarted loop is the score. Raw file: `benchmark/results/size-shared-stability.json`.

| Field | Value |
| --- | --- |
| finished | true |
| elapsed | 3607 seconds |
| loops | 323 |
| engine | coreml |

The hour completed with the app still running. Speed, sound, and stability all pass at 330.9 MB.

## S006 — Int8 pitch, decoder, and generator

Weight-only symmetric int8, per output channel. Activations and I/O remain
fp32. Each fixed shape was compressed separately because coremltools 9.0
rejects direct compression of multifunction packages; the compressed shapes
were then merged. The compiled sizes show that sharing survived.

| Stage | Shared fp32 compiled | Shared int8 compiled |
| --- | ---: | ---: |
| pitch | 33,959,547 | 8,892,606 |
| decoder | 134,687,955 | 34,151,789 |
| generator | 79,854,579 | 21,203,034 |

This approach is rejected. On the frozen 3s chain, pitch and energy were
59.5 dB and 62.3 dB, but the slight pitch change accumulated through harmonic
phase. Final waveform SNR fell to 2.5 dB. Package size alone is not enough.

## S007 — Int8 decoder and generator; pitch fp32

Text, prosody/duration, and pitch remain fp32. Decoder and generator use the
same weight-only int8 method as S006.

Frozen 3s waveform parity:

| Compression | Waveform SNR |
| --- | ---: |
| decoder only | 36.7 dB |
| generator only | 31.5 dB |
| decoder + generator | 30.8 dB |

The unsigned Release app is 171,696,317 bytes. The signed installable app is
171,739,063 bytes. This is under the 200 MB tier.

### iPhone 12 speed

Raw file: `benchmark/results/size-int8-dg-speed.json`.

| Case | Median RTF | Sample count | Thermal |
| --- | ---: | --- | --- |
| 3s | 0.185 | matched | fair |
| 7s | 0.211 | matched | fair |
| 15s | 0.223 | matched | serious |
| 30s | 0.269 | matched | serious |

Every length is under 0.30.

### iPhone 12 listening

Raw file: `benchmark/results/size-int8-dg-listen.json`.

| Case | Mark | Winner |
| --- | --- | --- |
| 3s | tie | neither |
| 7s | tie | neither |
| 15s | tie | neither |
| 30s | tie | neither |

Reference wins 0. The sound gate passes.

### iPhone 12 stability

Raw file: `benchmark/results/size-int8-dg-stability.json`.

| Field | Value |
| --- | --- |
| finished | true |
| elapsed | 3625 seconds |
| loops | 146 |
| engine | coreml |

The hour completed with the app still running. Speed, sound, and stability all
pass for the 171.7 MB candidate.

### iPhone 12 memory

Recorded separately with a 15 ms `task_vm_info.phys_footprint` sampler, so it
does not affect the graded speed numbers. This is the app process footprint and
may exclude private Neural Engine allocations. Raw file:
`benchmark/results/size-int8-dg-memory.json`.

| Case | Peak load | Peak synthesis | After release |
| --- | ---: | ---: | ---: |
| 3s | 73.7 MB | 422.4 MB | 176.5 MB |
| 7s | 221.2 MB | 752.3 MB | 216.0 MB |
| 15s | 419.1 MB | 1,074.8 MB | 139.6 MB |
| 30s part 1 | 410.9 MB | 1,144.9 MB | 150.6 MB |
| 30s part 2 | 425.0 MB | 1,123.3 MB | 156.5 MB |

Peak observed footprint was 1,144.9 MB. After three additional 30-second loops,
the footprint was 156.7, 155.8, and 155.5 MB, showing no growth in this short
diagnostic. Thermal state was serious throughout. The signed diagnostic-enabled
Release app is 171,819,479 bytes.

## S008 — 100 MB tier, rejected

Weight-only compression. Activations and inputs stay fp32. Pitch stays fp32.
No Release app was built, and the phone was not retested. The installed app is
still the 171.8 MB diagnostic build.

### Int8 text and prosody

Compiled multifunction packages, `weight_threshold=2048`, per-channel symmetric
int8:

| Stage | Compiled bytes | Compared with shared fp32 |
| --- | ---: | ---: |
| text | 6,733,080 | 0.2645 |
| prosody | 13,774,221 | 0.2566 |

Rounded durations on the five frozen chunks. A change rejects the candidate.

| Case | Int8 text | Int8 prosody | Both |
| --- | --- | --- | --- |
| 3s | 1/56, +1 frame | 0 | 0 |
| 7s | 1/110, +1 | 0 | 1/110, +1 |
| 15s | 3/237, +1 | 2/237, −2 | 7/237, −1 |
| 30s part 1 | 3/260, −1 | 0 | 3/260, −1 |
| 30s part 2 | 3/280, −3 | 5/280, −3 | 6/280, −4 |

Raising the prosody threshold to 250,000 still changes one real token in the
second 30-second chunk: index 190, token id 131, duration 2 → 3. Threshold
1,000,000 keeps every rounded duration. That one-shape package is 41,745,327
bytes against a 53,534,560-byte fp32 package, about 12 MB. Text does not have
a threshold that both shrinks and keeps durations. At 2,000,000 the text
package matches the fp32 package, so nothing is quantized. At 1,000,000 the
3-second clip is unchanged and the four longer clips are not.

### 4-bit decoder and generator

coremltools palettization allows a per-tensor table or a grouped-channel table.
It rejected per-channel. Group size 16 skipped the 1028- and 2180-channel
weights. Group size 4 divides those channels. One 1-channel weight stays fp32.

| Model | Bytes | Notes |
| --- | ---: | --- |
| Decoder, group 16, compiled | 40,322,853 | Larger than int8 (34,151,789) |
| Decoder, group 4, compiled | 37,023,241 | Still larger than int8 |
| Generator, group 4, compiled | 13,323,689 | Smaller than int8 (21,203,034) |
| Decoder, per-tensor, one shape | 16,987,835 | Not merged or compiled |

Grouped 4-bit saves about 8 MB on the generator and loses about 3 MB on the
decoder. The per-tensor decoder package is about 17 MB before compile. Using
that optimistic decoder, the duration-safe prosody cut, fp32 text, and fp32
pitch still leaves the app near 135 MB. The fp32 text, prosody, and pitch
models alone are 113.1 MB before the 3.2 MB app overhead.

At this point the 100 MB tier looked unreachable while rounded durations and
pitch stayed unchanged. S011 later went under 100 MB by accepting a larger
log-mel distance and judging the sound by listening. scikit-learn for the
palette clustering was installed under `artifacts/.pydeps` only.

## S009 — Selective weights, still above 100 MB

Duration LSTMs stay fp32. The prosody text-feature branch (embedding, three
convolutions, and its LSTM) is weight-only int8. Pitch keeps the F0 stack and
both output projections in fp32. The loudness convolutions before the final
projection are int8. No app was built.

| Piece | Compiled bytes | 3-second waveform |
| --- | ---: | --- |
| Prosody text branch int8 | 36,954,304 | 43.718 dB |
| Pitch loudness branch int8 | 26,501,773 | 43.268 dB |
| Pitch internal int8, including F0 convolutions | 13,559,524 | 7.342 dB, rejected |
| Decoder, one 4-bit table per tensor | 36,917,842 | 2.561 dB, rejected |
| Generator, one 4-bit table per tensor | 14,877,247 | 10.623 dB, rejected |
| Generator, 4-bit groups of 4 channels | 13,323,689 | 14.276 dB, rejected |

The text-branch prosody model changes 0 of the rounded durations on all five
frozen chunks. The rejected pitch model still reports 59.4 dB on the pitch
curve, and the harmonic source falls to 2.375 dB. The loudness-only pitch
model leaves that curve at 126.495 dB.

Replacing the current prosody and pitch models with the two passing pieces,
and keeping the int8 decoder and generator, projects a Release app of
147,484,184 bytes. Together, that full 3-second chain is 30.865 dB and changes
no duration. That is still above 100 MB, so the phone was not retested.

## S010 — Per-group sensitivity sweep

Each group was compressed alone first. Decoder and generator groups were then
tested as int4 inside the otherwise-int8 stage. The realistic 3-second chain,
not output-tensor SNR, decides the preliminary audio result.

### Text and prosody

| Group | Compiled bytes | Duration result | Chain result |
| --- | ---: | --- | --- |
| Text embeddings, int8 | 24,392,490 | exact, all five chunks | 12.432 dB, rejected |
| Text attention, int8 | 18,413,501 | 4 changed tokens | rejected |
| Text FFN, int8 | 16,050,252 | 10 changed tokens | rejected |
| Text output, int8 | 24,296,541 | 1 changed token | rejected |
| Text output, 8-bit k-means | 25,873,077 | exact | larger than fp32 |
| Prosody text branch, int4 | 34,156,234 | exact | 24.309 dB, rejected |

Exact rounded durations are insufficient. Quantized text embeddings alter
duration-hidden features, which alter F0; the harmonic source falls to 7.596
dB even though every rounded duration is unchanged.

### F0

Five groups were tested separately: front LSTM, style projections, and three
convolution blocks. Per-channel linear int8 produced 5.985–22.719 dB audio.
Eight-bit k-means produced −1.346–18.469 dB. Every group is rejected. The best
case was the last F0 convolution block with linear int8: pitch SNR 82.704 dB,
source-wave SNR 17.579 dB, and final-audio SNR 22.719 dB.

### Decoder and generator

Whole-stage linear int4 compiled to 17,369,455 bytes for the decoder and
11,392,268 bytes for the generator. Their isolated chain results were 11.263
and 10.025 dB.

With one int4 section and all other weights int8:

| Stage | Tested sections | Compiled range | Audio range |
| --- | --- | ---: | ---: |
| Decoder | encoder and four decode blocks | 30,076,028–32,437,402 | 15.998–23.612 dB |
| Generator | upsample, noise, early/late residuals, output | 17,074,251–21,203,045 | 11.252–31.527 dB |

The generator upsample case at 31.527 dB did not shrink: its compiled size is
effectively the same as the 21,203,034-byte int8 generator. All variants that
did shrink are rejected. Six-bit per-tensor k-means was also larger than int8:
115,346,072 bytes for the decoder and 41,190,736 for the generator.

No layer group in this sweep beat the 147,484,184-byte projection on waveform
SNR. S011 then changed the audio check to DTW log-mel plus listening and built
a smaller app.

## S011 — Sub-100 MB stack

Int8 text embeddings, int4 prosody text branch (duration LSTMs fp32), int8
pitch, int4 decoder and generator. Activations and I/O stay fp32. Built from
`benchmark/ios/project-s011.yml` (`Resources-S011/`, Swift flag `S011`), so the
S007 build is untouched.

### Durations

`check_selective_chain.py --combined-durations`: fp32 text → fp32 prosody
against candidate text → candidate prosody. Each chunk is checked in its own
process, because loading all ten models in one process segfaults coremltools.

| Case | Changed tokens | Frames |
| --- | ---: | ---: |
| 3s / t56 | 0/56 | 146 |
| 7s / t110 | 0/110 | 275 |
| 15s / t237 | 0/237 | 567 |
| 30s part 1 / t260 | 0/260 | 614 |
| 30s part 2 / t280 | 0/280 | 633 |

### 3-second chain

Waveform SNR is recorded only. DTW log-mel is the mean absolute dB per bin over
80 mel bins (1024 FFT, hop 256) along the DTW path
(`scripts/logmel_dtw.py`). Reference: PyTorch with the same captured noise.

| Chain | Waveform SNR | DTW log-mel |
| --- | ---: | ---: |
| fp32 Core ML | 43.397 dB | 0.018 dB |
| S007 int8 decoder + generator (validated) | 30.767 dB | 0.192 dB |
| **S011 candidate** | **2.028 dB** | **2.562 dB** |
| text + prosody only | 11.473 dB | 0.333 dB |
| pitch int8 only | 2.496 dB | 0.491 dB |
| decoder int4 only | 11.263 dB | 0.953 dB |
| generator int4 only | 10.025 dB | 2.346 dB |

Candidate: 87,600 samples, equal to the reference. Finite, peak 0.341, RMS 0.046.
Pitch's low SNR is mostly phase. The candidate's log-mel distance is 13× S007's,
and DTW barely lowers it (diagonal 2.596 dB). The int4 generator causes most of
it. Audio: `artifacts/3s_chain_s011_candidate.wav`.

### Size

| Build | Bytes |
| --- | ---: |
| Unsigned arm64 Release `.app` | 99,489,355 |
| Signed installable `.app` | 99,532,331 |
| Five compiled models | 96,203,053 |

The app contains only the five candidate `.mlmodelc`. Under 100,000,000 by
510,645 bytes unsigned.

### iPhone 12 speed

Raw file: `benchmark/results/size-s011-speed.json`.

| Case | Median RTF | Sample count | Thermal |
| --- | ---: | --- | --- |
| 3s | 0.171 | matched | nominal |
| 7s | 0.196 | matched | nominal |
| 15s | 0.213 | matched | nominal |
| 30s | 0.226 | matched | serious |

Every length is under 0.30. Speed passes.

### iPhone 12 listening

Raw file: `benchmark/results/size-s011-listen.json`.

| Case | Mark | Winner |
| --- | --- | --- |
| 3s | tie | neither |
| 7s | tie | neither |
| 15s | tie | neither |
| 30s | tie | neither |

Reference wins 0. The sound gate passes by the GOAL.md rule. Listener note: some
clips sounded slightly lower in volume ("not a lil noticeable"), but not enough
to prefer either side. Consistent with the 3s chain: peak 0.341 against 0.424,
RMS 0.046 against 0.049 (about −0.5 dB).

### Level, all five chunks

Follow-up to the listening note. `check_selective_chain.py --case`, same
reference and DTW (banded to ±64 frames; the 3s value matches the full DTW).

| Case | S011 RMS Δ | S011 peak / ref | S011 DTW log-mel | S007 RMS Δ | S007 DTW log-mel |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3s | −0.53 dB | 0.341 / 0.423 | 2.562 dB | +0.04 dB | 0.192 dB |
| 7s | −0.46 dB | 0.388 / 0.386 | 2.723 dB | +0.04 dB | 0.220 dB |
| 15s | −0.54 dB | 0.407 / 0.493 | 2.876 dB | +0.04 dB | 0.252 dB |
| 30s part 1 | −0.54 dB | 0.341 / 0.410 | 3.015 dB | +0.04 dB | 0.268 dB |
| 30s part 2 | −0.55 dB | 0.402 / 0.476 | 2.991 dB | +0.04 dB | 0.265 dB |

Every chunk keeps every duration and sample count. S011 is a steady ~0.5 dB
quieter in RMS, and peaks drop by up to 19% (about 1.8 dB) in four of five
chunks. S007 has no level change. Audio: `artifacts/<case>_chain_s011_candidate.wav`.

### Stability

Skipped. On 2026-10-16 the 60-minute loop was made optional for this build.
Speed and blind listening are the gates that were scored. The file named
`stability.json` on the phone is the earlier S007 hour, so it is not an S011
score.

## S012 — S011 with the generator level restored

Goal: recover the S011 level drop within the 510,645-byte margin.
`compress_sweep_variant.py` gains `mixed8` (int4 everywhere, named groups int8),
multi-group `--group`, and per-block groups `generator-res3/4/5` and
`generator-res5-p0/1/2` (res5 conv pairs, one per dilation).

### Where the level goes (3s chain, one stage changed, rest fp32)

| Change | RMS Δ | DTW log-mel |
| --- | ---: | ---: |
| text + prosody (S011) | +0.01 dB | 0.333 dB |
| pitch int8 | 0.00 dB | 0.491 dB |
| decoder int4 | +0.06 dB | 0.953 dB |
| generator int4 | −0.59 dB | 2.346 dB |
| generator int8 (S007) | +0.04 dB | 0.187 dB |
| int8 generator, output layer int4 | −0.21 dB | 1.735 dB |
| int8 generator, res3-5 int4 | −0.18 dB | 0.983 dB |
| int8 generator, noise / res0-2 / upsample int4 | +0.08 / −0.01 / +0.04 dB | 0.540 / 0.704 / 0.187 dB |

### Generator variants (int4, listed groups int8; generator alone)

| Kept int8 | Compiled bytes | Over int4 | RMS Δ | DTW log-mel |
| --- | ---: | ---: | ---: | ---: |
| none (S011) | 11,392,268 | 0 | −0.59 dB | 2.346 dB |
| output | 11,402,165 | 9,897 | −0.32 dB | 1.581 dB |
| output + res3 | 11,549,656 | 157,388 | −0.31 dB | 1.553 dB |
| output + res4 | 11,746,294 | 354,026 | −0.31 dB | 1.461 dB |
| output + res3 + res4 | 11,893,770 | 501,502 | −0.30 dB | 1.431 dB |
| output + res5 | 11,942,872 | 550,604 | −0.14 dB | 1.448 dB |
| output + res5 p0 / p1 / p2 | ~11,582,400 | ~190,150 | −0.25 / −0.26 / −0.28 dB | 1.543 / 1.526 / 1.540 dB |
| **output + res5 p0 + p1** | **11,762,659** | **370,391** | **−0.19 dB** | **1.486 dB** |

Output + res5 does not fit. Output + res5 p0 + p1 is the chosen generator.

### Full S012 stack, all five chunks

Same as S011 except the generator.

| Case | Durations | RMS Δ (S011) | Peak / ref | DTW log-mel (S011) |
| --- | --- | ---: | ---: | ---: |
| 3s | 0 changed | −0.11 (−0.53) dB | 0.372 / 0.423 | 1.705 (2.562) dB |
| 7s | 0 changed | −0.06 (−0.46) dB | 0.402 / 0.386 | 1.807 (2.723) dB |
| 15s | 0 changed | −0.08 (−0.54) dB | 0.423 / 0.493 | 1.960 (2.876) dB |
| 30s part 1 | 0 changed | −0.09 (−0.54) dB | 0.383 / 0.410 | 2.052 (3.015) dB |
| 30s part 2 | 0 changed | −0.10 (−0.55) dB | 0.470 / 0.476 | 2.033 (2.991) dB |

Sample counts match, all finite. Waveform SNR stays near 2 to −4 dB (phase).
Audio: `artifacts/<case>_chain_s012_candidate.wav`.

### Size

Built from `benchmark/ios/project-s012.yml` (`Resources-S012/`, Swift flag `S012`).

| Build | Bytes |
| --- | ---: |
| Unsigned arm64 Release `.app` | 99,861,698 |
| Signed installable `.app` | 99,904,916 |

138,302 bytes under 100,000,000 unsigned.

### iPhone 12 speed

Raw file: `benchmark/results/size-s012-speed.json`.

| Case | Median RTF | Sample count | Thermal |
| --- | ---: | --- | --- |
| 3s | 0.170 | matched | nominal |
| 7s | 0.198 | matched | nominal |
| 15s | 0.208 | matched | nominal |
| 30s | 0.221 | matched | fair |

Every length is under 0.30. Speed passes.

### Listening

No blind result recorded: the phone's `listen.json` was unchanged (same pair
IDs as S011), so no S012 pairs were marked. Informal listener note: acceptable,
with a very slight volume drop still audible. The sound gate is not scored
for S012. The app now writes the build name into `listen.json` as soon as
pairs are made, so a later install of the same bundle cannot be scored from
the previous file.

### Stability

Not run. S011 was later accepted without this loop. S012 itself has no new
blind listening score, so it is not the accepted 100 MB result.

## S013 — 50 MB tier, arithmetic floor

No model was compressed, compiled, or installed. Every number is measured from
the compiled S011 models already in `artifacts/` (`scripts/s013_size_floor.py`).
Per-weight bytes are the gaps between the unique `BLOBFILE` offsets in each
`model.mil`, so the five shared shapes count once.

### Protected pieces

Kept bytes that no tested compression leaves without a rounded-duration change
(S008 threshold probe; S010 text attention, FFN, and output groups):

| Piece | `weight.bin` bytes |
| --- | ---: |
| Text body: ALBERT layer, `bert_encoder` projection, biases, norms | 23,640,256 |
| Prosody duration path: duration LSTMs, AdaLayerNorm fc, `duration_proj` | 31,084,928 |
| **Protected total** | **54,725,184** |

Left out, so the floor is a lower bound: S011's int8 text embeddings
(367,232), the int4 prosody text branch (2,849,728), and the two `model.mil`
and metadata files (385,002 and 221,578).

### Floor

| Piece | Bytes |
| --- | ---: |
| Protected total | 54,725,184 |
| S011 pitch, int8 | 8,892,606 |
| S011 decoder, int4 | 17,369,455 |
| S011 generator, int4 | 11,392,268 |
| App overhead (S011 app 99,489,355 − its models 96,203,053) | 3,286,302 |
| **Minimum Release `.app`** | **95,665,815** |

That is 45,665,815 bytes over 50,000,000. The protected pieces alone are
4,725,184 over. After pitch, decoder, generator, and overhead, 9,059,369 bytes
are left for text and prosody, one sixth of what they need. The overflow comes
from both protected pieces; each is too large even alone:

| Case | Floor |
| --- | ---: |
| Without the text body | 72,025,559 |
| Without the duration path | 64,580,887 |
| Both protected pieces cast to fp16 (half size, not tested) | 68,303,223 |
| Without both | 40,940,631 |

### Structural reductions

Only removing both protected pieces gets under 50 MB. Nothing inside the
bundle does that:

- Byte-identical blobs of 4 KiB or more across the five models: 0 bytes. The
  ALBERT layer already stores one shared copy, and S002 already merged the
  fixed shapes.
- Merging text and prosody into one model saves only a `model.mil` and some
  metadata, under 0.6 MB, not any weights.
- Storing durations or text features for the five frozen chunks removes the
  models, but the app would then stop mapping token IDs to waveform. GOAL.md
  requires that, so this does not count.
- Downloading the text and prosody models after install would leave 40.9 MB in
  the bundle. GOAL.md says that downloaded models do not count, but it also
  says the app is offline. No such build exists.

The 50 MB tier is out of reach while rounded durations stay unchanged. No app
was built.

## S014 — Distilled front end, feasibility

Goal: replace the text body and prosody duration path with one small student
so the Release `.app` fits in 50,000,000 bytes. Pitch, decoder, and generator
stay the S011 models. The student, compiled metadata included, has
**9,059,369 bytes** (50,000,000 − 40,940,631 from S013).

### What the downstream needs

Read from `kokoro/model.py` `forward_with_tokens` and the S011 Swift chain
(`CoreMLChain.swift`). After text and prosody, the app uses three tensors:

| Output | Shape | Used by |
| --- | --- | --- |
| `duration_logits` | [1, T, 50] | `round(sum(sigmoid))`, clamp ≥ 1 → frames per token |
| `duration_hidden` | [1, T, 640] | aligned → pitch (F0 and energy). The last 128 values are the prosody style, copied |
| `text_features` | [1, 512, T] | aligned → decoder |

So the student only has to learn 50 + 512 + 512 values per token. The teacher
modules it replaces have 19,478,018 parameters (ALBERT 5,701,888 used,
`bert_encoder` 393,728, duration encoder 5,913,600, duration LSTM 1,839,104,
`duration_proj` 25,650, TextEncoder 5,606,400). The ALBERT pooler (590,592)
is unused.

### Student (`scripts/s014_student.py`)

    token ids [1,T] + af_heart style [1,256]
    x = embed(ids) + linear(style[128:])            # width 256
    8 pre-norm blocks, layout c c a c c a c c
        c: depthwise conv k=5 + feed-forward 256→512→256
        a: 4-head self-attention (no position table) + feed-forward
    duration_logits = linear(x) → 50       # same meaning as duration_proj
    duration_hidden = [linear(x) → 512, style[128:]]
    text_features   = linear(x) → 512

The outputs keep the S011 prosody model's names and shapes, so Swift drops the
text model and feeds `input_ids` and `style` to one model. There is no
length-dependent constant, so the five fixed shapes share every weight.

### Size, measured before training (`scripts/s014_size_probe.py`)

Untrained weights, fp16, five functions `t56`…`t280`, `save_multifunction`,
`coremlcompiler`. Weight bytes do not depend on weight values.

| Student | Parameters | Compiled `.mlmodelc` | `weight.bin` | Other | Projected app |
| --- | ---: | ---: | ---: | ---: | ---: |
| width 256 | 3,002,162 | 6,514,060 | 6,233,792 | 280,268 | 47,454,691 |
| width 192 | 1,760,370 | 4,034,592 | 3,754,304 | 280,288 | 44,975,223 |

Width 256 leaves 2,545,309 bytes of the budget; width 192 leaves 5,024,777.
Both fit, so the size gate does not stop the work. The app number is
projected; it is measured only after a real build.

### Data

- Corpus (`scripts/s014_corpus.py`): 44 public-domain Gutenberg books →
  77,483 utterances, 12,476,439 tokens, median 141, max 508.
  Neighbouring sentences are grouped to random lengths (10–460 characters).
  Phonemes come from espeak-ng `en-us` through `kokoro_onnx.Tokenizer`, the
  same call that made the frozen chunks; it reproduces the 3s, 7s, and 15s
  token ids exactly.
- Templates (`scripts/s014_templates.py`): 6,000 utterances, 789,714 tokens
  of numbers, dates, prices, phone numbers, questions, names, and lists, which
  old books lack.
- Validation (used for early stopping and tuning): 1,500 utterances, 257,522
  tokens from five other authors (London, Shaw, Kafka, Hesse, Franklin).
- Hidden test (`artifacts/s014/hidden_test/sentences.txt`, sha256
  `ec1902203acbb1013f38daa9d525586feba19d2469df85d8bb0546efcfbe8167`):
  120 hand-written sentences, 15 each short, question, number, name,
  punctuation, difficult phoneme; 10 each long, dialogue, list. No hidden
  sentence is a training or validation utterance. Six short phrases ("Yes.",
  "Who knew?") occur inside longer book utterances, and a few first names
  overlap the templates. The set is not used before the final candidate.
- The five frozen chunks are not in training; they are diagnostics only.

### Teacher targets (`scripts/s014_teacher.py`, `scripts/s014_cache_teacher.py`)

fp32 Kokoro, batched with packed LSTMs. Batched against one utterance at a
time: max 4.4e-4 absolute difference, no rounded duration changed. The cache
stores the continuous duration (fp32), and the 512 hidden and 512 text values
as per-token int8 (39.6 dB against the teacher). Holdout: 260,522 tokens in
55 s on MPS. Train (13.27 M tokens) is about 47 minutes, about 13.7 GB.

### Loss

    duration      L1 on sigmoid-sum duration, plus a term at rounding edges
    text_feature  MSE on text_features
    prosody       MSE on the 512 learned duration_hidden values
    pitch_energy  frozen fp32 F0Ntrain on student hidden, aligned with
                  teacher durations, against F0Ntrain on teacher hidden
    log-mel       frozen fp32 decoder + generator on 2 s crops, same noise
                  seed for student and teacher, multi-resolution log-mel

Stage A trains the first three losses. Stage B adds the last two.

### Compute (M4 Pro, 48 GB, MPS)

| Step | Measured | Estimate |
| --- | --- | --- |
| Teacher front end | 4,700 tokens/s | 47 min for train |
| Frozen pitch + decoder fwd+bwd, 8 × 2 s crops | 0.65 s/step | Stage B ~1 s/step |
| Stage A, 3 M student | — | 1–2 h |
| Stage B, ~20k steps | — | ~6 h |

### Gates for a built candidate

Superseded on 2026-10-17. Use `GOAL.md`. Speed and one blind listener are the
gates. The 60-minute loop, a 10–20 listener panel, MOS, and a 55% confidence
bound are not gates. The 120-sentence set may be used as source material. It
is not a required study. No listening had been run when this section was first
written.

### Blockers and open decisions

Resolved later in this section. Flexible frames were adopted. The listener
panel was dropped. The metallic edge was traced to the int4 generator.

1. **Sample counts with fixed shapes.** Pitch, decoder, and generator are
   fixed at 146, 275, 567, 614, and 633 frames, and `CoreMLChain.swift:82`
   throws unless the predicted frames add up exactly. A student that
   generalizes will not reproduce the teacher's rounded total exactly. With
   continuous error σ per token, the total drifts about σ·√T frames; even
   σ = 0.3 gives about ±4.6 frames on 237 tokens. Exact reference sample
   counts on all five chunks would need overfitting, which is not allowed.
   Both this and the 120 hidden sentences need pitch, decoder, and generator
   to accept any frame count (same weights and compression, flexible length).
   The gate would then be "sample count = 600 × the candidate's own frames".
   Decision needed.
2. **Listeners.** The 10–20 listener blind study needs people. I can prepare
   the randomized pairs and a blind A/B and MOS page. I cannot supply the
   listeners.
3. **Downstream under the stricter gate.** The S011 pitch, decoder, and
   generator passed only four clips with one listener, who noted a slight
   volume drop. A failure on 120 sentences could come from them, not the
   student. Separating the two needs a third arm: student + fp32 downstream.

### Feasibility verdict

Size, data, compute, and conversion are feasible. The width-256 student
compiles to 6,514,060 bytes, and the projected app is 47,454,691 bytes.
Naturalness is unknown until the student is trained. Decisions 1–3 are needed
before an iPhone build, not before training.

### Decisions, 2026-10-16

1. Pitch, decoder, and generator get a flexible frame length (same weights,
   same S011 compression). The sample-count gate becomes 600 × the candidate's
   own frames, on the five chunks and on every hidden sentence.
2. Listening has three arms: original 82M, student + fp32 downstream, and the
   full candidate.
3. Listening setup (page or WAV sheet) is decided when a candidate exists.
4. Width 256 is trained first; width 192 follows only if 256 passes.

### Training setup, 2026-10-16

- Teacher cache (`scripts/s014_cache_teacher.py`), fp32 teacher on MPS:
  - train: 83,483 utterances (77,483 book + 6,000 template), 13,433,119 tokens with pads, 1,977 s, 13 GB.
  - validation (holdout books): 1,500 utterances, 260,522 tokens.
  - Checks: token IDs match the corpus on 500 random train rows; no NaN; no zero scales; durations 0.96–28.7; int8 hidden SNR 39.6 dB.
- Trainer: `scripts/s014_train.py`.
  - Stage A loss = duration (L1 + rounding-edge term) + hidden-feature MSE + text-feature MSE, each normalized by per-channel teacher std.
  - Stage B adds pitch/energy (frozen F0Ntrain, L1 on voiced F0 / 100, L1 on N) and multi-resolution log-mel (FFT 512/1024/2048) through the frozen decoder and generator.
  - Stage B runs on 8 aligned 80-frame windows per step. The teacher and student see the same window and the same noise seed.
  - Model selection uses only the holdout-book validation split; the sealed hidden test is never read.
- Stage B step time: per-utterance version 4.5 s/step (GPU free); batched-window version about 1.5 s/step (while stage A was also using the GPU).
- Run `w256-a`: width 256, `ccaccacc`, 3,002,162 parameters, 40,000 steps, 8k-token batches, AdamW lr 1e-3 with cosine decay. Evaluated every 2,000 steps.

### Run w256-a (stage A), 2026-10-16

40,000 steps, 5,535 s on MPS, about 22 epochs. Best checkpoint = step 40,000 (`artifacts/s014/runs/w256-a/best.pt`).

Validation (holdout books, 1,500 utterances): duration exact 86.6%, frame MAE 0.167/token,
total-frames median error 1.37%, hidden SNR 6.37 dB, text SNR 9.11 dB. Train loss and
validation both flattened after about 20k steps: the student underfits; it is not overfitting.

Diagnostics (`scripts/s014_eval.py`, student front end + fp32 PyTorch downstream vs fp32 teacher, same noise seed):

| Set | Duration exact | Total-frames median err | F0 MAE (voiced) | Voicing agree | Energy MAE | DTW log-mel | Noise floor | RMS Δ | Clipped | Sample counts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 5 frozen chunks | 90.0% | 1.11% | 11.1 Hz | 98.5% | 0.238 | 3.80 dB | 0.72 dB | −0.06 dB | 0 | all = 600 × own frames |
| 100 validation | 86.8% | 1.36% | 9.9 Hz | 98.7% | 0.197 | 3.84 dB | 0.73 dB | +0.00 dB | 0 | all = 600 × own frames |

Noise floor = teacher against itself with a different noise seed. Waveform SNR (about −2.7 dB) is
phase-dominated and not informative. For scale: S011's downstream compression alone measured 2.56–3.02 dB
DTW log-mel with the same captured noise. The stage-A student alone is already further from the teacher
than that, so stage B, which trains on audio directly, has to close this gap. No WER: no offline ASR model is installed.

### Run w256-b (stage B), done, 2026-10-16/17

Started from w256-a best: lr 3e-4, `--w-total 10`, 8 windows × 80 frames per step, about 1.6 s/step.
Validation log-mel loss: 1.085 (stage A) → 0.650 (1k) → 0.638 (2k) → 0.618 (3k) → 0.600 (4k) → 0.583 (5k) → 0.555 (6k) → 0.545 (7k) → 0.535 (8k) → 0.529 (9k) → 0.528 (10k).
10,000 steps, 14,132 s. Still improving slowly at 10k; stopped there.

DTW log-mel against the fp32 teacher, same noise seed. The eval band is now 64 + the length
difference, after one validation utterance came out 30 frames short and gave inf:

| Checkpoint | Frozen: full | Frozen: features only | Frozen: durations only | Val (30): full | Val: features only | Val: durations only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| w256-a best | 3.80 | — | — | 3.84 (100) | — | — |
| w256-b step 2000 | 2.90 | — | — | 3.20 (29 of 30) | — | — |
| w256-b step 5000 | 2.89 | 2.40 | 2.19 | 3.13 | 2.43 | 2.52 |
| **w256-b step 10000** | **2.79** | **2.26** | **2.14** | **2.93 (100)** | **2.17** | **2.45** |

- Features only = student features on the teacher's durations.
- Durations only = teacher features on the student's durations.

Teacher features with the student's durations alone (about 12–15% of tokens one frame off) already
score 2.2–2.5 dB. Once timing changes, the decoder's noise and the pitch LSTM shift, and DTW does not absorb that.

Calibration correction: S011's 2.56–3.02 dB was measured with identical durations, so only the
"features only" column is comparable to it. The full-path number mixes in duration changes, and
DTW against the teacher cannot tell whether those changes are audible. Listening decides.

Step 10000 (100 validation utterances, a different random draw from the 30 above):
- Duration exact 86.3%, total length median error 1.4%.
- F0 MAE 9.2 Hz, voicing agreement 98.8%, energy MAE 0.175.
- RMS +0.07 dB, peak 0.408 vs 0.408, no clipping, all sample counts = 600 × own frames.
- Worst full-path DTW 3.97 dB.

**Chosen: w256-b step 10000** (`artifacts/s014/runs/w256-b/step10000.pt`). It is lower than step 5000 on every DTW column on the
frozen chunks, which are the same inputs for both. `best.pt` uses the stage-A feature score and is not used for stage B.

### Flexible downstream, 2026-10-17

`scripts/s014_convert_downstream.py` (one stage per process) re-converts pitch, decoder body and generator
from the same PyTorch wrappers, unchanged, with ct.RangeDim frames 1–2048 (2F and 120F+1 as their own RangeDims).
It applies the S011 compression: pitch int8, decoder and generator int4, per channel, linear symmetric, threshold 2048.
`scripts/s014_check_downstream.py` compares against the S011 fixed functions on CPU_ONLY. Report:
`artifacts/s014/downstream/report.json`.

| Model | Flexible compiled | S011 fixed compiled | Delta | Min SNR vs S011 (5 chunks) |
| --- | ---: | ---: | ---: | --- |
| pitch int8 | 8,622,252 | 8,892,606 | −270,354 | bit-identical |
| decoder int4 | 17,124,260 | 17,369,455 | −245,195 | 135.5 dB |
| generator int4 | 10,389,116 | 11,392,268 | −1,003,152 | 121.3 dB (phase) |
| **Sum** | **36,135,628** | **37,654,329** | **−1,518,701** | |

- F=401 and F=1500: all three load, run, give exact shapes, finite.
- Mac, ALL, F=633, warm median: pitch 17.7 vs 10.1 ms; decoder 20.4 vs 18.3 ms; generator 393.8 vs 393.4 ms.
- Budget for text/prosody replacement rises to 9,059,369 + 1,518,701 = **10,578,070 bytes**.
- Projected app with the w256 student (6,514,060 bytes, projection only; not yet converted): 47,454,691 − 1,518,701 = **45,935,990 bytes**.

### Student in Core ML, 2026-10-17

`scripts/s014_convert_student.py` converts w256-b step 10000 with input_ids [1, RangeDim(1, 512)] and style [1,256].
Settings: fp16 compute, fp32 I/O, iOS18, single function. `scripts/s014_check_student.py` compares it against the PyTorch fp32 student.
Report: `artifacts/s014/student_coreml/report.json`.

- Compiled: **6,076,253 bytes** (rechecked with `logical_bytes`), 3,002,162 params. The fixed-shape probe was 6,514,060.
- Frames vs PyTorch: 99.8% of tokens identical (frozen 941/943; holdout 16,797 tokens). 24 of 100 holdout utterances differ by
  1–2 frames in total. CPU_ONLY and ALL give identical results.
- SNR vs PyTorch: duration_hidden ≥ 47.9 dB, text_features ≥ 46.4 dB.
- Shapes pass at T = 1…512. Mac ALL latency 0.4 ms (T=32) to 3.6 ms (T=512).

Projected app = 3,286,302 overhead + 36,135,628 downstream + 6,076,253 student = **45,498,183 bytes**,
4,501,817 under 50,000,000. The projection is under the limit, so building the S014 app is allowed.

### S014 app build, 2026-10-17

`project-s014.yml` + `Resources-S014/` (student + three flex packages; no S011 text/prosody), Swift changes behind `#if S014`:
- CoreMLChain: one student call replaces text + prosody; F = the student's own frame sum drives every shape.
- The fixed-frame guard is dropped; the sample guard is `audio.count == 600 × F`.
- Bench: RTF and audioSeconds come from the produced audio.
- Memory/ContentView: the sample flag checks a multiple of 600, because the throwing guard in `synthesize` is the real check.

Unsigned arm64 Release `.app`: **45,489,655 bytes** (rechecked), 4,510,345 under 50,000,000.

| Inside the app | Bytes |
| --- | ---: |
| decoder int4 flex | 17,124,260 |
| generator int4 flex | 10,389,116 |
| pitch int8 flex | 8,622,252 |
| student w256 | 6,076,253 |
| Everything else | 3,277,774 |

S011 still builds. Not yet run on the iPhone.

### S014 on the iPhone 12, 2026-10-17

Two device-only bugs, fixed in Swift under `#if S014`:
1. The student's outputs are FLOAT16 in the spec (the downstream models' are FLOAT32). `values()` read them as Float,
   so the 3s chunk gave 769 frames instead of 143. It now converts fp16 arrays.
2. The flexible generator failed with `.all` and with CPU-only on the phone. The error was "E5RT: BNNS convolution: Error in
   shape deduction: dilated kernel size should not be smaller than padded input". The Mac CPU_ONLY was fine. Fix: load the generator with `.cpuAndGPU`.

The temporary error and frame-count logging was removed afterwards. Unsigned Release `.app` after the fixes: **45,492,639 bytes**
(+2,984 bytes of Swift).

Speed, `-coreml-speed`, 2 warmups + 10 scored runs per chunk, RTF on the student's own audio. `benchmark/results/size-s014-speed.json`:

| Case | Median RTF | p90 | Max | Own audio (s) | Samples = 600 × own frames | Thermal | S011 median |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 3s | 0.178 | 0.182 | 0.183 | 3.575 | yes | fair | 0.171 |
| 7s | 0.199 | 0.201 | 0.204 | 6.800 | yes | fair | 0.196 |
| 15s | 0.203 | 0.204 | 0.204 | 14.175 | yes | serious | 0.213 |
| 30s | 0.252 | 0.269 | 0.280 | 31.250 | yes | serious | 0.226 |

Speed gate passes (≤ 0.30 at every duration). The phone was already warm from 6 install/run cycles, so 30s ran at "serious".
This speed run used the build with the diagnostic logging; the only change since is removing file writes outside the timed models.

### S014 blind listening test set-up, 2026-10-17

- **Done:**
  - Built the in-app blind A/B test: the "Hidden listening (S014)" screen on the iPhone 12.
  - Made all clips for the 120 sealed hidden sentences. The hidden set is used only for listening, never for tuning.
- **How:**
  ```
  hidden (sha256 checked) -> espeak en-us -> ids (pad 0 both ends)        # scripts/s014_listen_prep.py ids
  Mac:    ref = fp32 Kokoro; student = w256-b step10000 + fp32 downstream  # seed 1000+index, no normalization
  iPhone: -make-listen -> S014 Core ML chain -> Documents/listen-out/s014  # app can't write into devicectl-copied dirs
  session(listener k) = 24 consecutive sentences from a fixed shuffle at 24k; all pairs ref vs s014;
                        ref on side A in 12 of 24 (seeded by k); prefer A/B/tie + MOS 1-5 for A and B
  ```
- **Result:**
  - All 120 iPhone clips pass the checks: length exactly 600 × frames, no NaN, peak at most 0.607, none silent or under 0.5 s.
  - |RMS difference| vs ref: mean 0.12 dB, max 0.45 dB.
  - Phone frame counts equal the Mac student's on 108 of 120 clips and are at most 1 frame off on the rest.
  - the owner listened to the Mac ref and student clips unblinded. His sessions count as a pilot only.
- **Next:** this panel was not run. Sound is one listener, per `GOAL.md`.

### S014 pilot listening and the metallic edge, 2026-10-17

- **Done:**
  - the owner ran one pilot session of 24 pairs. It is pilot-only because he had already heard the Mac clips.
  - Located the cause of the artifact.
- **Result:**
  - Original preferred 18 times, S014 0 times, tie 6 times. MOS: original 4.88, S014 4.25.
  - the owner heard "a high pitch, metallic, sharp sound" over clear speech.
- **Diagnosis:**
  ```
  DTW log-mel (30 hidden clips): ref-student 3.0 dB, student-s014 4.3 dB, ref-s014 5.0 dB
  generator flex on Mac CPU vs GPU vs ALL: 113-120 dB SNR (GPU path not the cause on Mac)
  band energy vs total (8-10k / 10-12k): ref -17.1/-18.9, student -17.2/-19.0, s014 -19.4/-22.6,
                                         S011 3s candidate -19.2/-22.5 (fp32 anchor -19.1/-21.2)
  ear test: 3s_chain_s011_anchor_fp32 vs 3s_chain_s011_candidate -> same metallic edge, "a bit"
  ```
  - The cause is the shared compressed downstream (int4 decoder and generator), not the student.
  - S011's 4-clip tie missed it. The 120-sentence A/B exposes it.
- **Budget:**
  - The app is 45,492,639 bytes, so 4.5 MB is left, plus about 2.6 MB of bench wavs that could leave the bundle.
  - Generator to int8 adds about 9.3 MB; decoder to int8 adds about 16.5 MB.
  - Neither fits whole.
- **Next:** decided in the following section. One targeted precision fix, inside 50 MB.

### S014 vocoder isolation and mixed8 generator, 2026-10-17

- **Done:**
  - the owner allowed one targeted mixed-precision exception within 50 MB, with no broad sweep.
  - Isolated the stage causing the artifact, then built the S012 mixed8 recipe on the flexible generator.
- **How:**
  ```
  scripts/s014_vocoder_isolate.py: 5 frozen chunks + 10 validation utterances (never the hidden set)
    student front end -> its durations -> fp32 F0Ntrain -> decoder (fp32 | Core ML) -> same sine
    spectrum -> generator (fp32 | Core ML) -> iSTFT; each arm vs fp32. One package set per process
    (four loaded Core ML models segfault).
  mixed8 flex = linear_quantize int4 per-channel, weight_197 (output) + res5 p0/p1 (175/179/183/187) int8
                (same const names and shapes as the fixed S012 generator, 101/101)
  ```
- **Result** (mean over 15 utterances, vs fp32; band energy relative to total):

  | Arm | DTW log-mel | RMS Δ | 6–8k | 10–12k |
  | --- | ---: | ---: | ---: | ---: |
  | fp32 | 0 | 0 | −20.97 | −20.28 |
  | int4 decoder + fp32 generator | 0.84 | +0.07 | −21.01 | −20.25 |
  | fp32 decoder + int4 generator | 2.57 | −0.58 | −19.80 | −22.08 |
  | int4 decoder + int4 generator (S014) | 2.69 | −0.52 | −19.81 | −22.05 |
  | fp32 decoder + mixed8 generator | 1.60 | −0.14 | −20.53 | −19.63 |
  | int4 decoder + mixed8 generator (candidate) | 1.77 | −0.07 | −20.54 | −19.58 |
  | int8 generator, reference only (+9.8 MB, does not fit) | 0.21 | +0.04 | −20.98 | −20.46 |

  - Generator sizes (compiled): int4 10,389,116; mixed8 10,759,434 (+370,318); int8 20,199,216.
  - Projected app with mixed8: about 45,862,957 bytes.
- **Next:** done in the following section. Mixed8 was built, and the iPhone listen passed.

### S014 mixed8 build, speed and blind listening, 2026-10-17

- **Done:**
  - Re-read GOAL.md. The sound gate is one blind listener, the four lengths plus a few unseen sentences. Panel, MOS, the 60-minute loop and confidence intervals are recorded, not graded.
  - Rebuilt S014 with the mixed8 generator (`kokoro_generator_mixed8_flex`, `.cpuAndGPU`), ran speed on the iPhone 12, remade the 120 phone clips, and ran blind listening.
- **How:**
  ```
  Resources-S014: int4 generator -> mixed8 generator; xcodegen; Release build, unsigned for size
  speed: -coreml-speed, 2 warmups discarded, median of 10 per length
  clips: -make-listen -> Documents/listen-out-m8 -> s014_listen_prep.py check s014-m8
  listen: HiddenListenView 8 unseen pairs (entries 24+, after the int4 pilot's 0-23)
          + Listen section, 4 lengths; sides random; listener-1
  ```
- **Result:**
  - **Size:** 46,107,717 bytes, 3,892,283 under 50,000,000. PASS.
  - **Speed** (warm RTF median / max; sample count = 600 × own frames on every run). PASS:

    | Length | Median | Max |
    | --- | ---: | ---: |
    | 3s | 0.173 | 0.188 |
    | 7s | 0.181 | 0.184 |
    | 15s | 0.189 | 0.193 |
    | 30s | 0.197 | 0.208 |

    Thermal state reached serious during the 30s run.
  - **Phone clips:** 120 of 120 OK; RMS vs reference +0.43 dB mean, 0.83 max; peak 0.716; frames match the Mac student on 108 of 120 (max difference 1).
  - **Sound:** 12 blind pairs, 12 ties, 0 reference wins, 0 candidate wins (`benchmark/results/size-s014-m8-listen.json`). No metallic edge reported. MOS (recorded, not graded): equal on every hidden pair. PASS.
  - **Tier:** S014 mixed8 passes speed, sound and size: **Research (≤ 50 MB)**. The size cut came from replacing the text body and duration path with the width-256 student; the one targeted fix was int8 on the generator output and res5 p0/p1 weights (+370,318 bytes).
- **Not run:** 60-minute loop and listener panel, skipped by the owner on 2026-10-17. Neither is a gate.

## S015 — Distilled downstream under 20 MB

Goal: an offline Release `.app` of at most 20,000,000 bytes, with the same
speed gate (warm median RTF ≤ 0.30 on the iPhone 12) and the one-listener
blind sound gate. S014 stays unchanged as the reference result; no S014 file
was modified. Naturalness comes first after the size limit: metrics below
are diagnostics, and only a blind listen against original Kokoro decides.

### Production budget (`scripts/s015_budget.py` → `artifacts/s015/budget.json`)

| Piece | Bytes |
| --- | ---: |
| S014 app | 46,107,717 |
| KokoroSize binary (includes diagnostic UI, upper bound) | 812,600 |
| Info.plist + PkgInfo | 930 |
| af_heart voice table, fp32 [510, 1, 256] | 522,240 |
| **Required non-model bytes** | **1,335,770** |
| Bench / listening only (3s, 7s, 15s, 30s-0, 30s-1 wavs + bench.json), left out of production | 2,711,988 |
| **Max compiled-model bytes** | **18,664,230** |
| S014 student (kept), compiled | 6,076,253 |
| **Max downstream with the S014 student** | **12,587,977** |
| S014 downstream (pitch int8, decoder int4, generator mixed8) | 36,505,946 (23,917,969 over) |

Runtime path kept: token ids + af_heart style row → S014 student → durations
→ pitch → sine source spectrum (Swift `SourceDSP`) → vocoder → iSTFT
(n_fft 20, hop 5) → 24 kHz, 600 samples per frame.

### Student downstream (`scripts/s015_downstream.py`)

Two Core ML models replace three, with the same tensor interfaces, so the
Swift sine source and iSTFT stay as they are:

```
PitchStudent   aligned [1,640,F] + style[128:]  -> pitch, energy [1,2F]
  conv 640->256, BiLSTM 128x2 (teacher 256x2), F0 and N branches 256/128 (teacher 512/256)
VocoderStudent text [1,512,F] + pitch, energy + style[:128] + spectrum [1,22,120F+1] -> magnitude, phase
  decoder width 256 (teacher 1024), output 256 ch (teacher 512)
  generator 256 -> 128 -> 64 ch (teacher 512 -> 256 -> 128), same Snake AdaIN resblocks
  (k 3/7/11, dilations 1/3/5), same sine-source injection, same 22-ch iSTFT head
params: pitch 2,506,242 (teacher 8,425,474), vocoder 8,394,726 (teacher decoder+generator 53,313,586)
```

### Size probe before training (`scripts/s015_size_probe.py`, untrained, int8 per-channel, flexible F 1–2048)

| Model | Compiled bytes |
| --- | ---: |
| Pitch | 2,646,608 |
| Vocoder | 8,874,117 |
| Downstream | 11,520,725 (budget 12,587,977) |
| **Projected app** | **18,932,748** (1,067,252 under) |

Flexible runs at F = 1, 146, 633, 1500: shapes right, outputs finite.
Mac warm latency at F = 633: pitch 8 ms, vocoder 138 ms (CPU+GPU); S014's generator alone was 394 ms.

### Training (`scripts/s015_train.py`)

```
data: S014 cache: train = 77,483 book + 6,000 template utterances; validation = 1,500
      holdout utterances by five other authors. Frozen chunks and the hidden set unused.
per step: 8 random train utterances; inputs from the fp32 teacher cache or (50%) from the
          S014 student (its hidden, text and durations), so the downstream sees app-like input
  pitch:   full utterances (InstanceNorm is over the whole sequence)
           loss L1(f0)/100 + L1(energy) vs fp32 F0Ntrain on the same input
  vocoder: one crop of 80/96/112/128 frames per utterance, teacher F0/N as input,
           one shared sine spectrum (same RNG seed) for teacher and student
           -> teacher pre-iSTFT logits (verified equal to model.decoder, max diff 0.0)
           loss = multi-res log-mel + multi-res STFT (SC + log-mag, 2048..128)
                  + 0.5 L1 log-magnitude logits + complex-spectrum L1
AdamW 4e-4, betas 0.8/0.99, warmup 300, cosine to 5%, grad clip 5
eval every 400 steps: 8 fixed holdout groups x {teacher, S014} inputs, crop 96
stop on NaN, divergence (> 2x best), 4 stale evals, < 20 GB free disk
```

Failures fixed before the run:
- NaN gradients on MPS from step 1. `torch.istft` backward was replaced by a
  real-valued overlap-add iSTFT (`scripts/s015_istft.py`, max diff 8.9e-8 vs
  `torch.istft`; trimmed before dividing by the window envelope, which is 0
  at the first untrimmed sample).
- Still NaN on MPS only (CPU clean): spectral losses (`torch.stft`) moved to
  CPU. 60 MPS steps clean after that.
- Speed: about 6–8 s per step on MPS (pitch teacher and student per
  utterance about 1.6 s, backward about 1.2 s, variable shapes). The run was
  restarted from step 400 with a 3,000-step cosine to fit the night.

### Run v1, 2026-10-17 (`artifacts/s015/runs/v1`)

- 3,000 steps (400 + resume to 3,000), 9,287 s after the resume, MPS. No NaN, no divergence, no early stop; validation improved at every eval:

  | Step | 400 | 800 | 1,200 | 1,600 | 2,000 | 2,400 | 2,800 | 3,000 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | Val score | 3.604 | 2.855 | 2.684 | 2.457 | 2.367 | 2.268 | 2.217 | 2.202 |

  Final train terms (last 100 steps): f0 0.067, energy 0.176, mel 0.707, mrstft 1.074, logmag 0.275, complex 0.254. Teacher-cache and S014-student inputs score the same (2.20 both ways).
- Checkpoint kept: `step3000.pt` (= `best.pt`).

### Evaluation (`scripts/s015_eval.py`, PyTorch fp32, vs original fp32 Kokoro)

100 holdout utterances (five unseen authors); the 5 frozen chunks agree within 0.1–0.4 dB.

| Arm vs fp32 Kokoro, same seed | DTW log-mel dB | RMS Δ dB | 6–8k | 8–10k | 10–12k |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp32 Kokoro, other seed (noise floor) | 0.73 | | | | |
| S015 downstream, teacher front end and durations | 3.15 | −0.59 | −0.73 | −0.89 | −0.73 |
| S015 full chain (S014 student + S015) | 3.59 | −0.59 | −0.56 | −0.84 | −0.75 |
| S014 student + fp32 downstream | 2.93 | +0.07 | +0.10 | −0.08 | −0.05 |

- S015 downstream on teacher input: F0 MAE 8.7 Hz, voicing agreement 0.988, energy MAE 0.19.
- All arms finite, non-silent, no clipping, samples = 600 × own frames on every utterance.
- The S015 downstream adds about 3.2 dB to the teacher path. The S014 int4 generator added 2.57 dB (fp32-relative) and was heard as metallic; mixed8 added 1.60 and passed. By that yardstick S015 v1 is further from Kokoro than an arm that already failed the ear.

### Core ML (`scripts/s015_convert.py`, `artifacts/s015/coreml/v1-step3000`)

| Model | Compiled bytes |
| --- | ---: |
| `kokoro_s015_pitch_int8` | 2,667,742 |
| `kokoro_s015_vocoder_int8` | 8,984,400 |
| Downstream | 11,652,142 |
| + S014 student 6,076,253 + required non-model 1,335,770 | |
| **Projected app** | **19,064,165** (935,835 under 20,000,000) |

- int8 vs PyTorch fp32 on the frozen chunks: F0 MAE 0.11–0.13 Hz, energy MAE 0.003; vocoder SNR 38.9–41.1 dB, DTW 0.10–0.12 dB. Quantization is not the problem; the student is.
- Core ML end to end: finite, samples = 600 × frames, peak 0.27–0.34.
- Mac warm latency (median of 5) at 605 / 644 frames: pitch 4.4–4.9 ms; vocoder 115 / 123 ms CPU+GPU, 169 / 189 ms CPU. S014's generator alone was 394 ms at 633 frames. Not measured on the iPhone.

### Diagnostic clips (`scripts/s015_listen_prep.py`, `artifacts/s015/listen/v1-step3000`)

- The 120 hidden-set token ids from `artifacts/s014/listen/ids.json` (the text was not read), S014 student → S015 Core ML int8, seed 1000 + index, 16-bit.
- DTW log-mel vs `artifacts/s014/listen/ref` (16-bit file vs 16-bit file): S015 4.14 dB; Mac S014 student + fp32 downstream 3.02 dB. RMS −0.88 dB, peak 0.28, no clipping, all finite and sample counts right.
- Measurement bug found and fixed: comparing float audio to a 16-bit reference inflated DTW (4.91), because float silence sits under the −80 dB log-mel floor while quantization noise does not.
- Not listened to. These are not a candidate; they exist only to let a listener hear the gap.

### Verdict

- Size: **fits**. The two-model downstream gives a projected 19,064,165-byte app.
- Quality: **not credible yet**. On objective diagnostics, S015 v1 is further from original Kokoro than the S014 int4 generator, which already failed a blind listen. The HF bands are 0.7–0.9 dB low (dull, not bright), and pitch is 8.7 Hz off on voiced frames.
- Per the task, **the S015 app was not built**: no `project-s015.yml`, no `Resources-S015/`, no Swift change. The README is unchanged. S014 mixed8 stays the result.
- 20 MB is not shown to be infeasible. It is unproven: v1 was one architecture, 3,000 steps, regression losses only.

### Next (not run)

1. Train much longer: tens of thousands of steps (validation was still falling at 3,000). At about 3–4 s per step on MPS, cache teacher F0/N per utterance first to remove the per-step teacher pitch pass.
2. Add HiFi-GAN-style adversarial and feature-matching losses (MPD + MSD) after the regression warm start. Regression-only vocoders are known to be over-smoothed, which matches the dull HF here.
3. Initialize from the teacher: slice the generator's top-norm channels, rather than starting from random weights.
4. About 0.94 MB of headroom remains: widen the last generator stage (64 → 80 channels) only if (1) and (2) plateau.
5. Then the phone: speed on the iPhone 12, and a blind listen against original Kokoro on unheard hidden entries 32+.

### MPS sine source skews v1 metrics, 2026-10-17

Found while porting the trainer to CUDA (A100 VM). `s015_train.evaluate` on v1 `step3000.pt`, same code, same seeds:

| Device | Val score | Full-chain DTW dB | RMS Δ dB | 6–8k | 8–10k | 10–12k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MPS (as reported for v1) | 2.202 | 3.11 | −0.64 | −0.92 | −0.77 | −0.49 |
| CPU | 2.385 | 3.55 | −1.31 | +0.29 | +0.54 | +0.48 |
| CUDA (A100) | 2.385 | 3.56 | −1.32 | +0.29 | +0.57 | +0.52 |

(Full chain here = 16 fixed holdout utterances, S014 student front end, S015 vs fp32 Kokoro downstream.)

- Teacher and student networks agree CPU vs MPS given the same sine spectrum `har` (max diff ≤ 1.5e-4); the deterministic sine part agrees too.
- CPU networks fed the MPS-drawn `har` reproduce the MPS score exactly (2.2021); fed the CPU `har`, 2.3848. So the difference is in the random source (`m_source` noise/phase) drawn on MPS, even though its mean magnitudes match (voiced 0.0899 vs 0.0900) and `randn_like` statistics look normal on both (std 1.000, no channel or lag correlation; MPS shows more duplicate values, unique fraction 0.85 vs 0.98). Root cause inside the MPS draw not pinned down.
- Consequence: v1 was trained and evaluated with MPS sources, so its reported numbers (val 2.20; eval table above, if run with `--device mps`) are optimistic against CPU / CUDA / Core ML. The CPU-true v1 step-3000 numbers are val 2.385, full-chain DTW 3.55. From here on every S015 number is CPU or CUDA; the S014 comparison draws its noise on CPU.
- Not yet checked: what the Swift SourceDSP noise looks like relative to either.

### S015 vs S014 mixed8 comparison (`scripts/s015_compare_s014.py`, `artifacts/s015/compare/v1-step3000`)

Same 15 items as the S014 isolation baseline (5 frozen chunks + holdout `default_rng(7).choice(…, 10)`), S014 student front end and durations, CPU only, sine seed 0 for every arm (asserted identical to the reference's captured spectrum). "voc" = fp32 F0/N given; "full" = the arm's own pitch model, sine source rebuilt from its F0 with the same seed.

| Arm vs fp32 Kokoro | DTW dB | RMS Δ dB | 6–8k | 8–10k | 10–12k | F0 MAE Hz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32, seed 1 (noise floor) | 0.404 | +0.00 | −0.00 | −0.00 | +0.02 | 0.00 |
| S014 mixed8 voc (int4 decoder + mixed8 generator) | **1.771** | −0.07 | +0.44 | −0.18 | +0.71 | 0.00 |
| S014 mixed8 full (+ int8 pitch) | **1.862** | −0.07 | +0.45 | −0.19 | +0.70 | 0.13 |
| S015 v1 voc, PyTorch fp32 | 3.267 | −1.40 | +0.64 | +0.93 | +0.66 | 0.00 |
| S015 v1 full, PyTorch fp32 | 3.459 | −1.39 | +0.63 | +0.91 | +0.65 | 7.91 |
| S015 v1 voc, Core ML int8 | 3.275 | −1.41 | +0.65 | +0.96 | +0.67 | 0.00 |
| S015 v1 full, Core ML int8 | 3.463 | −1.40 | +0.64 | +0.94 | +0.64 | 7.90 |

- S014 mixed8 voc reproduces the 1.771 dB baseline exactly, so the harness matches `s014_vocoder_isolate.py`.
- S015 v1 is 1.5–1.6 dB further from Kokoro than S014 mixed8 on both views, 1.4 dB quiet, and its F0 is 7.9 Hz off on voiced frames. The bar for an S015 build is ≤ 1.86 dB full / ≤ 1.77 dB voc, with no level or band regression.

### Run v2-a100 set-up, 2026-10-17 (GCP `<gpu-vm>`)

- VM: <gpu-machine-type>, one A100-SXM4-40GB (driver 580.178.04), 12 vCPU, 83 GB RAM, image pytorch-2-9-cu129-ubuntu-2204. Fresh venv (`--system-site-packages`): Python 3.10.12, torch 2.9.1+cu129, CUDA 12.9, cuDNN 9.10.2, numpy 2.2.6, kokoro 0.9.4, misaki 0.9.4, transformers 4.46.3.
- Transferred: scripts, `models/pytorch`, `voices-v1.0.bin`, S014 student `w256-b/step10000.pt`, v1 `step3000.pt` / `last.pt` / log / args, `benchmark/inputs`, S014 cache holdout (257 MB) and train (13,877,602,780 bytes, 7 files, SHA-256 identical to the Mac copy). Upload about 4 MB/s. A stopped `rsync --partial` left a truncated `hidden_scale.npy` (24.3 of 26.9 MB); caught by the size/checksum check and re-sent.
- Pitch cache (`s015_cache_pitch.py --variant {teacher,s014}`, two processes in parallel): 77,661 utterances per variant, 61.7 M / 61.6 M values, 25 min, exact fp32. Spot check vs a live teacher pass: identical (teacher inputs), max 0.0002 Hz (S014 inputs).
- Probes: resume from step 3000 OK, startup eval 2.38487 (= the CPU value), all losses finite with the adversarial path on, checkpoints and logs written. 100-step benchmark with adversarial + FM at full weight: **0.87 s/step**, peak 21.9 GB allocated (25.4 GB used), GPU 88–98 % busy with dips while batches are built. Estimated 17,000 steps ≈ 4.1 h.
- Gradient norm at the restart was 42 (v1's last was about 11), falling to 17 within 100 steps: v1 had learned MPS-drawn sources (see above); train mel 0.81 → 0.78.
- Feature-matching weight raised from the planned 0.2 to **1.0** before launch: after 100 discriminator steps FM was 0.094, so 0.2 × FM would have been under 1 % of the reconstruction loss (about 2.3), i.e. absent. Adversarial stays 0.1 (3.8 → about 16 %). Discriminators from 3,000, losses reach the students from 4,000, linear ramp to full by 8,000.
- Launched 13:41 CDT in tmux `v2`: `s015_train_gan.py --name v2-a100 --init artifacts/s015/runs/v1/last.pt --steps 20000 --eval-every 2000 --patience 3 --fm-weight 1.0 --device cuda`.

### Run v2-a100 result, 2026-10-17 (`artifacts/s015/runs/v2-a100`)

- 3,000 → 20,000 (17,000 steps) in 13,870 s (3 h 51 min), 0.80–0.82 s/step on the A100. Stopped at `--steps`: no NaN, no divergence, no stale evaluation, disk ≥ 54 GB free throughout, 0 pitch-cache misses.
- Validation (CUDA, exact fp32 eval; full chain = 16 fixed holdout utterances, S014 student front end):

  | Step | 3,000 | 4,000 | 6,000 | 8,000 | 10,000 | 12,000 | 14,000 | 16,000 | 18,000 | 20,000 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | Val score | 2.385 | 2.275 | 2.143 | 2.036 | 1.955 | 1.905 | 1.847 | 1.806 | 1.780 | **1.766** |
  | Chain DTW dB | 3.556 | 3.297 | 3.091 | 2.993 | 2.851 | 2.813 | 2.716 | 2.674 | 2.618 | **2.611** |
  | Chain RMS Δ dB | −1.32 | −0.48 | −0.52 | −0.10 | −0.04 | −0.36 | −0.09 | −0.07 | −0.10 | −0.08 |
  | 10–12k Δ dB | +0.52 | −1.59 | −1.45 | −0.68 | −0.50 | −0.47 | −0.06 | −0.41 | +0.09 | −0.03 |

  Gains were slowing at the end (−2.2 %, −1.4 %, −0.8 % per eval) but had not plateaued under the 0.5 % rule. Final train terms (last 100 steps): f0 0.040, energy 0.106, mel 0.581, mrstft 0.894, logmag 0.200, complex 0.171; disc 3.86, adv 2.15, FM 0.62.
- Best checkpoint: `step20000.pt` (= `best.pt`, by validation score and by chain DTW). `last.pt` keeps discriminators, optimizers, schedulers and history for a resume.
- `s015_eval.py --device cuda`, 100 holdout utterances, step 20,000 (step 18,000 marginally worse on nearly every term):

  | Arm vs fp32 Kokoro, same seed | DTW dB | RMS Δ dB | 6–8k | 8–10k | 10–12k |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | fp32 Kokoro, other seed (noise floor) | 0.43 | | | | |
  | S015 v2 downstream, teacher front end | 2.64 | −0.05 | −0.23 | −0.21 | |
  | S015 v2 full chain (S014 student + S015) | 3.42 | −0.03 | −0.10 | −0.15 | −0.18 |
  | S014 student + fp32 downstream | 2.92 | +0.07 | +0.09 | −0.07 | −0.13 |

  F0 MAE 5.74 Hz (v1 8.75), voicing agreement 0.993, energy MAE 0.127. (The v1 table above was an MPS run; its noise floor read 0.73.)
- **Against S014 mixed8** (`s015_compare_s014.py`, the 15-item 1.77 dB set, CPU, identical inputs and sine noise; `artifacts/s015/compare/v2-a100-step20000`):

  | Arm vs fp32 Kokoro | DTW dB | RMS Δ dB | 6–8k | 8–10k | 10–12k | F0 MAE Hz |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | fp32, seed 1 (noise floor) | 0.404 | +0.00 | −0.00 | −0.00 | +0.02 | 0.00 |
  | S014 mixed8 voc | **1.771** | −0.07 | +0.44 | −0.18 | +0.71 | 0.00 |
  | S014 mixed8 full | **1.862** | −0.07 | +0.45 | −0.19 | +0.70 | 0.13 |
  | S015 v1 voc / full (PyTorch) | 3.267 / 3.459 | −1.40 / −1.39 | +0.64 | +0.93 | +0.66 | 0 / 7.91 |
  | S015 v2 voc (PyTorch fp32) | 2.310 | −0.11 | −0.23 | +0.10 | +0.06 | 0.00 |
  | S015 v2 full (PyTorch fp32) | 2.537 | −0.13 | −0.24 | +0.10 | +0.08 | 5.03 |

- Verdict: v2 closed about 60 % of v1's gap (full 3.46 → 2.54 dB; level and HF bands now within ±0.25 dB, flatter than S014 mixed8's +0.7 dB at 10–12k), but it is still **0.68 dB further from Kokoro than S014 mixed8 on the full downstream, and 0.54 dB on the vocoder alone**. The rule was "build only when S015 is at least as good as S014 mixed8 objectively", so **no S015 build and no blind ear test**. S014 mixed8 stays the result; README unchanged. Not converted to Core ML (v1's int8 matched PyTorch within 0.01 dB, and the PyTorch model already fails the bar).
- 20 MB is still not shown infeasible: validation was still improving at 20,000 steps.
- Copied back and verified (79 files, SHA-256 identical to the VM, 1.4 GB): all `stepN.pt`, `best.pt`, `last.pt`, `log.jsonl`, `args.json`, `env.json`; `vm-logs/` (trainer, cache and eval logs, pip list, nvidia-smi, probe and benchmark args/logs, pitch-cache meta); `eval/` (100-utterance evals of steps 18,000 and 20,000). The 13.8 GB S014 cache and the 1 GB pitch cache stayed on the VM. VM stopped 17:47 CDT (22:47Z in `reports/gcp-vm-ledger.md`; status TERMINATED, not deleted). Wall time created → stopped about 6 h.

## S015 final vocoder stage width: untrained size probes (2026-10-17, Mac only, VM stopped)

`VocoderStudent` gained an optional `last` (final generator stage channels; default unchanged at
gen // 4 = 64, and `config` only records it when set, so v1/v2 checkpoints load as before).
`s015_eval.load_downstream` now builds the vocoder from the checkpoint's `vocoder_config`.
`s015_size_probe.py vocoder --last C` writes `artifacts/s015/probe/size_vocoder_int8_lastC.json`
(the v1 probe files are untouched). Untrained weights, int8 per-channel, compiled with coremlcompiler.

| final stage | parameters | package bytes | compiled bytes |
| ---: | ---: | ---: | ---: |
| 64 (v1/v2) | 8,394,726 | 8,851,634 | 8,874,117 |
| 80 | 8,965,126 | 9,434,546 | 9,457,036 |
| 96 | 9,633,830 | 10,109,951 | 10,132,474 |

Projected app = pitch + vocoder + S014 student 6,076,253 + required non-model 1,335,770:

| final stage | untrained (pitch probe 2,646,608) | with v1's trained drift (+131,417) | vs 20,000,000 |
| ---: | ---: | ---: | --- |
| 64 | 18,932,748 | 19,064,165 (measured, v1) | under by 935,835 |
| 80 | 19,515,667 | ~19,647,084 | under by ~352,916 |
| 96 | 20,191,105 | ~20,322,522 | **over**: ruled out |

The drift is v1's measured trained-minus-untrained compiled size (vocoder +110,283, pitch +21,134);
a trained 80 build must still be measured. 88 channels would land near 19.98 MB with drift, too close.
No training started.

## Run v3-w80 set-up: 80-channel final stage, warm-started from v2 (2026-10-17)

Plan (owner): v2 body kept; the final stage widened 64 -> 80; v2's 64 channels copied; the extra 16
from Kokoro's 128-channel final stage, skipping channels that duplicate ones v2 already has;
10,000 steps on the A100, evaluate every 2,000; stop if chain DTW > 2.3 dB from 6,000 steps;
measure the compiled app from the first real checkpoint and stop above 20,000,000 bytes;
build and ear test only at about 1.86 dB on the 15-item S014 comparison.

Graft (`scripts/s015_widen.py`, Mac CPU, 47 s; train-split crops only, 8 fit + 4 held-out
batches of 8 utterances, crop 96, half teacher and half S014-student inputs):
- Teacher channels ranked by activation std x conv_post column norm. A channel is a duplicate if
  its activation |corr| with any v2 channel (or an already picked one) is above 0.5.
  42 stronger teacher channels were skipped as duplicates; picked
  [1, 105, 39, 117, 86, 69, 73, 91, 40, 38, 18, 29, 82, 5, 28, 67], max |corr| to v2 0.30–0.46.
- Per-channel tensors (biases, AdaIN rows, Snake alphas, noise_convs rows) and the new x new
  blocks of the resblock convs come from the teacher (weight norm folded). Old <-> new blocks
  are zero, so v2's channels are not changed by the new ones.
- ups.1 new outputs cannot be copied (teacher stage-0 is 256 channels, v2's is 128), so they were
  fitted to the teacher's ups.1 output on the picked channels: R² 0.744 fit, 0.737 held out.
- conv_post new columns, held-out log-mag L1: zero 0.2082, teacher x0.25 0.2112, x0.5 0.2196,
  x1 0.2490, ridge fit to the residual 0.2421 -> zero. The new channels have teacher-like
  activations but start with no output weight; Adam moves those columns from the first step.
  Asserted: with zero columns the widened model reproduces v2's logits (max diff < 1e-4).
- 15-item comparison of `init.pt` (s015-voc): DTW 2.310, RMS −0.11, the same as v2.
- `artifacts/s015/runs/v3-w80/{init.pt, graft.json}`. Parameters: vocoder 8,965,126.

Trainer (`s015_train_gan.py`): `--init` builds the vocoder from the checkpoint's `vocoder_config`,
loads AdamW moments only if saved (fresh here, the widened tensors changed shape) and loads the
discriminators + their optimizer when saved (v2's here). New `--gate STEP:DTW` stops at the first
evaluation from STEP whose chain DTW (16 rng-11 holdout utterances) is above DTW.
The comparison and eval loaders build the vocoder from `vocoder_config`, so v1/v2 load unchanged.

VM: the A100 zone was STOCKOUT at 23:0xZ; retried every 3 min, started 23:36Z (new IP). Pushed the
four scripts and init.pt (SHA-256 matched). Launch:

    .venv/bin/python scripts/s015_train_gan.py --name v3-w80 --init artifacts/s015/runs/v3-w80/init.pt \
        --steps 30000 --gate 26000:2.3 --lr 1e-4 --lr-d 1e-4 --fm-weight 1.0 --device cuda

Peak lr 1e-4 (v2 used 2e-4 from v1): the body is already tuned and the student moments are
fresh, so the restart bump is kept smaller. 500-step warmup, cosine to 5% at 30,000; adversarial
and FM losses at full weight from the start (v2's discriminators). Step 20000 eval on the VM:
score 1.7657, chain DTW 2.611, the same as v2. About 0.85 s/step.

### v3-w80 first real checkpoint (step 22000)

Eval: score 1.8331 (start 1.7657), chain DTW 2.699 (start 2.611), RMS −0.26: the expected restart bump
(fresh student AdamW, lr warmup to 1e-4).
Compiled size (`s015_convert.py artifacts/s015/runs/v3-w80/step22000.pt`, int8, trained weights):
pitch 2,667,742 + vocoder 9,569,103 = 12,236,845; + S014 student 6,076,253 + non-model 1,335,770
= **19,648,868 bytes**, under 20,000,000 by 351,132 (projection was ~19,647,084). PyTorch vs Core ML int8
on the frozen chunks: vocoder DTW 0.19–0.23 dB, SNR 34.7–35.4 dB, all finite. Size gate passed; run continues.

### v3-w80 step 24000

Eval: score 1.7571 (v2 best 1.7657; under the 0.5% bar, stale 2), chain DTW 2.613 (v2 2.611), RMS −0.08.
15-item S014 comparison (CPU, NOISE_SEED 0, `artifacts/s015/compare/v3-w80-step24000/`):
s015-voc 2.289 (v2 2.310), s015-full 2.520 (v2 2.537; F0 MAE 4.87 Hz). Target for a build: ~1.86.

### v3-w80 result: gate stop at step 26000

| step | score | chain DTW (16 rng-11 utts) | chain RMS Δ |
| ---: | ---: | ---: | ---: |
| 20000 (= v2) | 1.7657 | 2.611 | −0.08 |
| 22000 | 1.8331 | 2.699 | −0.26 |
| 24000 | 1.7571 | 2.613 | −0.08 |
| 26000 | 1.7156 | 2.511 | +0.01 |

Stop: `gate: chain DTW 2.511 > 2.3 at step 26000` (5230 s). The score improved 2.8% over v2.

15-item S014 comparison (CPU, NOISE_SEED 0), DTW dB vs fp32 Kokoro:

| arm | v2 step20000 | v3 step24000 | v3 step26000 | S014 mixed8 |
| --- | ---: | ---: | ---: | ---: |
| vocoder (fp32 F0/N) | 2.310 | 2.289 | 2.201 (RMS −0.03) | 1.771 |
| full (own F0/N) | 2.537 | 2.520 | 2.436 (F0 MAE 4.94 Hz) | 1.862 |

The new channels are used. conv_post column norms (median) for new vs old channels: 0.068 vs 0.334 at
22000, 0.112 at 24000, 0.132 at 26000, still rising. The gap to S014 full went from 0.675 to 0.574 dB.
That is about 0.05 dB per 2,000 steps after the restart bump, so the remaining 4,000 steps would
not reach ~1.86. No build, no ear test. S014 mixed8 stays the result; v2 step20000 stays the baseline
(v3 step26000 is the best S015 checkpoint by both metrics, 19,648,868 bytes projected at step 22000).

Files on the Mac (SHA-256 matched with the VM): `artifacts/s015/runs/v3-w80/{init.pt, step20000–26000.pt,
best.pt (= 26000), last.pt (with disc + optimizers), log.jsonl, args.json, env.json, graft.json,
vm-logs/v3-w80.log}`; compare JSONs in `artifacts/s015/compare/v3-w80-{init,step24000,step26000}/`.
VM stopped 2026-10-18 01:08Z (ledger updated); A100 time this session about 1.5 h (23:36Z–01:08Z).

## S015 v4-prune: vocoder pruned from Kokoro, then distilled

Plan (agreed 2026-10-18): prune Kokoro's fp32 decoder and generator down to the v3 shape (width 256, gen 256,
asr 32, last 80, i.e. the same 19,648,868-byte projected app). Select channels, refit by least squares,
then distill for 10,000 steps: reconstruction only for steps 0–2000, then ramp GAN and feature matching over
2000–3000, with v3's discriminators. The trainer gate stops the run from step 6000 if the chain DTW is above
v3's 2.511, the trainer counterpart of the 15-item 2.44. Close rule, on the 15-item vocoder arm: close the
20 MB tier if the end value is ≥ 2.05 dB **and** the last 2,000 steps improved it by < 0.05 dB.

### Init: select + refit (`scripts/s015_prune.py`, Mac CPU, 301 s)

- Mapping: student decode.0 ← T.decode.0; decode.1 ← T.decode.1 norm1 plus T.decode.2 convs/norm2,
  refitted to T.decode.2's output; decode.2 ← T.decode.3 (upsample).
- Channel score: activation std on train crops × norm of the teacher weights that consume the channel.
  Top-k: 256/1024 per decoder stream, 256/512 at the upsample, asr 32/64, generator 128/256 and 80/128.
- Per-channel tensors (AdaIN, instance norm, Snake alphas, depthwise pool, noise_convs rows) are sliced
  exactly.
- Every conv with a pruned input gets a ridge least-squares refit (1e-3) in forward order, on the student's
  own activations (6 fit batches of 8 utts, train split, seed 303). Decode shortcuts fit
  √2·teacher block output − student residual. ups.0/1: Adam MSE, 300 iterations. conv_post fits the
  teacher logits.
- Per-layer R²: decoder 0.79–0.99 (sliced −0.29 to 0.87); generator 0.67–1.00; conv_post 0.984.

Held-out log-mag L1 vs teacher logits (2 batches, seed 404):

| | L1 |
| --- | ---: |
| sliced only | 1.282 |
| sliced + refit (v4 init) | 0.318 |
| v3-w80 step26000 (trained) | 0.196 |

15-item comparison of the init (CPU, NOISE_SEED 0), vocoder arm: **6.467 dB** DTW, RMS −3.41 dB,
6–8k −1.31 / 8–10k −3.99 / 10–12k −3.58 dB (v3 step26000: 2.201). The refitted init is quiet and dull:
chained least-squares fits shrink toward the mean, and the cut drops high-band channels. This is a starting
point, not a result; the reconstruction phase has to repair it. JSON in `artifacts/s015/compare/v4-prune-init/`.

### Run v4-prune (A100, launched 2026-10-18 ~04:40Z)

VM started 04:35Z after 11 STOCKOUT retries (ledger row written). init.pt SHA-256 `234b0de4…a91e178`, the same
on the Mac and the VM.
`s015_train_gan.py --name v4-prune --init …/v4-prune/init.pt --steps 10000 --lr 2e-4 --lr-d 1e-4
--disc-start 2000 --adv-start 2000 --adv-ramp 1000 --adv-weight 0.1 --fm-weight 1.0 --eval-every 2000
--gate 6000:2.511`. Fresh student AdamW, v3 discriminators loaded. Step-0 eval: chain DTW 6.675, RMS −3.36.

| step | score | chain DTW (trainer) | chain RMS Δ | 15-item vocoder arm | 15-item full arm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (init) | — | 6.675 | −3.36 | 6.467 | — |
| 2000 (recon only) | 1.6346 | 2.457 | −0.44 | 2.078 (RMS −0.44, 8–10k −1.02, 10–12k −0.75) | — |

Step 2000 already beats v3 step26000 on every measure (score 1.716, chain 2.511, vocoder arm 2.201).
| 4000 (GAN full from 3000) | 1.6282 | 2.460 | −0.03 | 2.139 (RMS −0.02, 8–10k +0.76, 10–12k +0.78) | — |

At 4000 the level is right, but the high band overshoots (it was −1 dB at 2000, now +0.8 dB): the GAN ramp moved it.
The vocoder arm got worse by 0.06 dB; the chain is flat (2.457 → 2.460).

Rule added 2026-10-18 (user): at 6000, if the run is better than v3 but no longer falling, finish the 10,000;
the end-of-run close rule decides.
| 6000 | 1.5219 | 2.256 | −0.30 | **1.946** (RMS −0.36, 8–10k +0.36, 10–12k +1.04) | **2.214** (F0 MAE 4.86 Hz) |

6k check: the chain is 2.256, under the 2.511 gate, and the full arm is 2.214, under v3's 2.436, so the run continues.
The vocoder arm, 1.946, is already under the 2.05 close line. Gaps to S014 mixed8: vocoder 0.175 dB (v3: 0.43), full
0.352 dB (v3: 0.574). The 10–12k band still overshoots by about 1 dB.
| 8000 | 1.4109 | 2.099 | −0.07 | **1.742** (RMS −0.09, 6–8k +0.11, 8–10k +0.13, 10–12k +0.01) | **2.038** (F0 MAE 4.70 Hz) |

At 8000 the vocoder arm (1.742) is below S014 mixed8's 1.771 on the same 15 items. The high band is flat
again. The full arm (2.038) is 0.176 dB from S014's 1.862. Comparing the arms, the vocoder now accounts for
about 0 of that gap; it comes from the unchanged v3 pitch student (F0 MAE 4.70 Hz). These are metrics only:
no build, no ear test.
| 10000 | 1.3658 | 2.037 | −0.13 | **1.671** (RMS −0.15, 6–8k +0.15, 8–10k +0.18, 10–12k +0.26) | **1.965** (F0 MAE 4.53 Hz) |

### v4-prune result: run complete at step 10000 (7907 s); the 20 MB tier stays open

Close rule (vocoder arm ≥ 2.05 **and** last gain < 0.05): the end value is 1.671 and the last 2,000 steps gained
0.071 dB, so the tier is **not** closed on either condition. The run was still falling at the end, since the
lr had decayed to its floor.

| 15-item DTW dB vs fp32 Kokoro | vocoder arm | full arm |
| --- | ---: | ---: |
| v2 step20000 | 2.310 | 2.537 |
| v3-w80 step26000 | 2.201 | 2.436 |
| **v4-prune step10000** | **1.671** | **1.965** |
| S014 mixed8 (reference) | 1.771 | 1.862 |

- The vocoder arm is 0.10 dB *below* S014 mixed8. The full arm is 0.103 dB above it; that gap is the unchanged v3
  pitch student, not this vocoder. It does not reach the ~1.86 full-arm bar for a build and ear test.
- Compiled size of step10000 (`s015_convert.py`, int8): pitch 2,667,742 + vocoder 9,569,102 = 12,236,844;
  projected app **19,648,867** bytes (351,133 under 20,000,000).
- Core ML int8 vs the fp32 student, on the convert chunks: vocoder SNR 29.7–31.7 dB, DTW log-mel 0.21–0.26 dB;
  all outputs finite. Warm vocoder time on 30 s chunks is 141–147 ms on CPU+GPU and 204–216 ms on CPU (Mac).
- Metrics only: no build, no listening, no parity claim. S014 mixed8 stays the result until a blind test.
- The init-from-teacher pruning is what worked: v3 added random capacity on top of v2, which did not help.

Files on the Mac (SHA-256 matched with the VM): `artifacts/s015/runs/v4-prune/{init.pt, prune.json, step0–10000.pt,
best.pt (= 10000), last.pt (with disc + optimizers), log.jsonl, args.json, env.json, vm-logs/v4-prune.log}`;
compare JSONs in `artifacts/s015/compare/v4-prune-{init,step2000,step4000,step6000,step8000,step10000}/`.
VM stopped 2026-10-18 06:54Z (ledger updated); A100 time about 2.3 h (04:35Z–06:54Z).

## S015 v5-pitch: retrain the pitch student against the frozen v4-prune vocoder

Agreed 2026-10-18. The vocoder already beats S014 on the vocoder arm, so only the pitch student trains.
Same PitchStudent shape (compiled 2,667,742 bytes, so the projected app stays 19,648,867 with a 351 KB
margin). Init: `v4-prune/step10000.pt` with step reset to 0 (`v5-pitch/init.pt`, SHA-256 `4c851fac…`).

`s015_train_gan.py --freeze-vocoder` (new flag): the vocoder is frozen and there are no discriminators.
- Loss: f0 + energy against cached teacher targets, plus mel + mrstft + 0.5·logmag + complex. The last four
  compare the frozen vocoder on the student's own F0/N with fp32 Kokoro on the teacher's F0/N (same crop,
  same sine seed), mirroring the s015-full arm.
- Kokoro's sine source runs under no_grad, so the audio losses reach the pitch student only through the
  vocoder's F0/N input convs.
- Each evaluation runs the 15-item S014 comparison (s015-full arm, CPU subprocess). It decides best.pt,
  staleness (no new best, patience 3) and the `--target-dtw 1.86` stop. The trainer's independent 16-utt
  chain is logged too, as a cross-check: stopping on the comparison set picks the best of several looks at
  it, which is optimistic.
- No build or listening unless it reaches ≤ 1.86.

v5-pitch not run: all 40 A100 start attempts in <gcp-zone> failed with STOCKOUT (06:59Z–09:04Z, every 3 min).
The VM stays stopped (TERMINATED, no start ledger row). Per the overnight plan, no other zone or VM was tried;
everything is ready to launch as soon as capacity returns. Staleness for this run = no new best on the
15-item full arm.

### v5-pitch run (2026-10-18, A100, the owner started the VM after the access grants were renewed)

Prep: `coremltools==9.0` installed into the VM venv (the comparison script imports it at the top; the
s015-full arm does not use it). No code changed. A standalone VM precheck of `init.pt` on the 15-item
s015-full arm gave 1.965 dB / F0 MAE 4.53 Hz, the same as the Mac.

`--steps 8000 --freeze-vocoder --eval-every 1000 --target-dtw 1.86 --lr 1e-4 --device cuda`, about 0.39 s/step.

| step | 15-item full dB | F0 MAE Hz | 16-utt chain dB | eval teacher_f0 | eval teacher_energy | stale |
|---|---|---|---|---|---|---|
| 0 | **1.965** | 4.53 | 2.037 | 0.0395 | 0.1072 | 0 |
| 1000 | 2.028 | 5.12 | 2.082 | 0.0433 | 0.1213 | 1 |
| 2000 | 1.999 | 5.10 | 2.065 | 0.0435 | 0.1194 | 2 |
| 3000 | 1.985 | 4.86 | 2.055 | 0.0415 | 0.1186 | 3 |

Stopped at step 3000 after three evaluations with no new best. The trainer prints "no 0.5% validation
improvement"; that message text is stale. In frozen mode, any new best counts. best.pt = step 0 (the
unchanged v3 pitch student). **Result: failed.** No candidate, no build, no listening; README unchanged.

Reading: every trained step is worse than the start on all three measures (15-item dB, F0 MAE, and the
pitch student's own held-out F0/energy loss against the teacher). The worse score at step 1000 matches the
learning-rate restart. It recovered toward the start by step 3000 but never passed it. Adding the audio losses
through the frozen vocoder did not tighten F0 to the teacher; it loosened it. The only route from those
losses to the pitch student is the vocoder's f0_conv/n_conv (the sine source is under no_grad), and
that route seems to reward F0/N values that suit the student vocoder, not teacher-accurate F0. The 1.965 →
1.862 gap (S014 full) is therefore still open at 20 MB.

Files pulled to `artifacts/s015/runs/v5-pitch/` (step0–3000.pt, best.pt, last.pt, log.jsonl, args.json,
env.json, `vm-logs/v5-pitch.log`); SHA-256 of all .pt/.json/.jsonl match the VM. VM stopped 17:23Z (ledger
start 16:50Z, stop 17:23Z; A100 time about 0.55 h).

## S015 v6-pitch-only: pitch student on teacher F0/energy only, vocoder frozen

Agreed 2026-10-18 after v5-pitch. v5's audio losses pulled F0 away from Kokoro (F0 MAE 4.53 → 5.12 Hz,
full chain 1.965 → 2.028 dB at step 1000). A wider pitch model was ruled out: it would cost vocoder width,
the vocoder already beats S014, and the margin is only 351 KB.
- Init: `v5-pitch/init.pt` (= v5 step 0: v3 pitch student + v4-prune step10000 vocoder, frozen).
- Loss: f0 + energy against the cached teacher targets only (`--pitch-loss-only`, new flag). lr 1e-5 (v5: 1e-4).
- 15-item s015-full comparison every 1000 steps decides best/staleness. A lower F0 error counts only if
  that full-chain number falls.
- Stop: ≤ 1.86 dB, three evaluations without a new best, or about 30 minutes (`--max-minutes 30`, new flag:
  evaluates the current weights, then stops).
- No build or listening unless ≤ 1.86. If it stays above 1.86, S014 remains the result.
- The trainer's stop message now reads "no new best" in frozen mode (the v5 log line said "0.5%").

### v6-pitch-only run (2026-10-18)

A100 start: STOCKOUT in <gcp-zone> from 19:00Z (3-min retries, then 5-min after one transient gcloud crash
at 20:10Z), until it started at 20:55Z. While waiting, the same run was also started on the Mac CPU (see below).

Trainer change for the Mac: `PitchCache` is optional. Without `artifacts/s015/cache/pitch/teacher`, every
utterance is a cache miss and the teacher runs live (the same path a miss already took). On MPS the trainer's
own validation fails (`stft input and window must be on the same device`), so the Mac used `--device cpu`.
Mac 20-step probe (`runs/v6-mac-probe`): step 0 1.965 dB, about 1.2 s/step, about 80 s per evaluation.

VM run `v6-pitch-only` (`--steps 20000 --freeze-vocoder --pitch-loss-only --eval-every 1000 --target-dtw 1.86
--lr 1e-5 --max-minutes 30 --device cuda`, cached targets, about 0.15 s/step):

| step | 15-item full dB | F0 MAE Hz | 16-utt chain dB | stale |
|---|---|---|---|---|
| 0 | 1.965 | 4.53 | 2.037 | 0 |
| 1000 | 1.965 | 4.55 | 2.036 | 1 |
| 2000 | **1.964** | 4.53 | 2.034 | 0 |
| 3000 | 1.970 | 4.53 | 2.037 | 1 |
| 4000 | 1.966 | 4.50 | 2.036 | 2 |
| 5000 | 1.965 | 4.52 | 2.032 | 3 |

Stopped at step 5000: no new best in 3 evaluations (1498 s, inside the 30-min cap). best.pt = step 2000,
1.964 dB. Held-out teacher_f0 0.0395 → 0.0393, essentially unchanged.

Mac run `v6-pitch-only-mac` (CPU, same flags with `--max-minutes 75`, live teacher): step 0 1.965, step 1000
1.966, step 2000 1.964. It matched the VM, so it was stopped by agreement at step 2,2xx. Checkpoints for steps
0, 1000 and 2000 are kept in `runs/v6-pitch-only-mac/`. Its last.pt is from the step-2000 evaluation.

**Result: failed.** Pitch and energy losses alone at lr 1e-5 leave the full chain at 1.964–1.970 dB and F0
MAE near 4.5 Hz, well above the 1.86 bar. With v5 (audio losses made it worse), retraining this pitch
student does not close the 1.965 → 1.862 gap. No build, no listening, README unchanged.
**S014 mixed8 stays the accepted result** (agreed 2026-10-18).

Files: `artifacts/s015/runs/v6-pitch-only/` (step0–5000.pt, best.pt, last.pt, log.jsonl, args.json,
env.json, `vm-logs/v6-pitch-only.log`); SHA-256 of all .pt/.json/.jsonl match the VM. VM stopped 21:24Z
(ledger start 20:55Z, stop 21:24Z; A100 time about 0.5 h).

## Next decision: S016 teacher-derived pitch model

Do not continue fine-tuning the S015 pitch checkpoint. Both tested objectives have closed that route:
v5's audio losses made the full chain worse, while v6's direct F0/energy losses left it flat.

The next bounded experiment is a new pitch model initialized from Kokoro's `F0Ntrain` weights using
structured channel selection and layer-wise refitting, following the method that succeeded for the
v4-prune vocoder. Keep the v4-prune step-10000 vocoder frozen and keep the same external pitch interface.
Before a long run:

1. Convert an untrained/refitted candidate to int8 Core ML and reject it if the projected app exceeds
   20,000,000 bytes. The current projection is 19,648,867 bytes, leaving only 351,133 bytes.
2. Measure step-0 teacher F0/energy error and the same 15-item full-chain comparison.
3. Train only a candidate that improves on the v6 starting point. Stop after three evaluations without
   a new 15-item best.
4. Require full-chain DTW log-mel <= 1.86 dB before building or preparing a listening test.

S014 mixed8 remains the accepted result unless all size, speed, and sound gates pass.

## S016 attempt 1: Kokoro F0Ntrain pruned to half width, layer-wise refit (2026-10-18)

VM: the user asked to keep `<gpu-vm>` (A100, <gcp-zone>) running. It was
restarted 21:27Z (ledger start row written) and is idle; nothing in this attempt ran on it.

**Architecture.** `PitchStudent(width=256, lstm=128, inp=False)` is Kokoro's predictor `F0Ntrain` layout at
half width: BiLSTM 640→128×2, then per branch (F0, N) AdainResBlk1d 256→256, 256→128 (upsample),
128→128, and proj 128→1. 2,735,362 parameters. The 640-dim aligned input goes straight to the LSTM (no
`inp` conv). Size probe (`s015_size_probe.py --pitch-config '{"width":256,"lstm":128,"inp":false}'`):
compiled int8 pitch 2,872,329 bytes, below the 3,018,875 cap. Projected app 19,853,454 bytes.
Code changes: `PitchStudent` takes `inp`/`half`. The config is saved in checkpoints as `pitch_config` and read
by `s015_eval.load_downstream`, `s015_train_gan.py` and `s015_size_probe.py`.

**Initialization** (`scripts/s016_prune_pitch.py`):
- Weight norm is folded. Every tensor is sliced from the fp32 teacher.
- Channels are ranked by activation std at each consumer × the consumer's weight-column norm, and the top k
  are kept. For the LSTM that is the top 128 hidden units per direction, with gate rows sliced to match.
- Identity-shortcut blocks keep the same channel set on input and output.
- The layers are then refit in forward order on the student's own activations. The LSTM is fit with Adam (200
  iterations, MSE to the teacher's LSTM output on the kept channels). Every conv is fit with ridge least
  squares (1e-3). A conv that closes a block is fit to √2·teacher block output minus the student's other
  branch. Proj is fit to the teacher's F0/N.
- Data: 48 fit and 16 held-out train-split utterances (seeds 303/404), half with teacher durations and
  half with S014 durations. The rng-7 comparison items and the hidden set were not used.

    python scripts/s016_prune_pitch.py --name s016-prune --vocoder artifacts/s015/runs/v4-prune/step10000.pt \
        --reference artifacts/s015/runs/v6-pitch-only/best.pt
    python scripts/s015_compare_s014.py s016-prune-init --arm s015-full --checkpoint artifacts/s015/runs/s016-prune/init.pt

Refit R² per layer (sliced → refit): shared 0.369→0.989; F0 branch 0.831→0.978, 0.892→0.989, 0.766→0.992,
0.993→0.997, conv1x1 0.793→0.925, 0.401→0.928, 0.626→0.881, proj 0.949→0.989; N branch all ≥0.976 after
refit. Run time 711 s on the Mac CPU (the LSTM fit took 680 s of that).

| pitch model | held-out F0 MAE vs teacher (Hz) | held-out energy MAE | 15-item full chain dB | 15-item F0 MAE (Hz) |
|---|---|---|---|---|
| sliced only | 54.39 | 3.096 | — | — |
| sliced + refit (init.pt) | 6.81 | 0.206 | 2.235 | 8.39 |
| v6-pitch-only best (reference) | 3.82 | 0.099 | 1.964 | 4.53 |

**Result: the initialization fails the gate.** It is worse than the 4.53 Hz / 1.964 dB baseline. Per-layer R²
is high, but errors compound through the three F0 blocks. Not converted, not trained. The user chose to revise
it with an end-to-end refit (attempt 2). If that also failed, the next try would be option 3 (move width from N to F0), done once.

Files: `artifacts/s015/runs/s016-prune/init.pt` (SHA-256 824f5c97…a4c6bd; pitch + frozen v4-prune
step-10000 vocoder) and `prune.json` (6020405e…7d5be; kept channels, per-layer R², held-out scores).
Log: `artifacts/s015/s016-prune.log`. Script SHA-256 ad091586…607543.

## S016 attempt 2: end-to-end refit of the attempt-1 init (2026-10-18)

The user's constraints:
- Train on the training split only; never on the 15 evaluation items.
- Optimize all pruned pitch weights jointly; keep the channel selections and architecture fixed.
- Loss: final teacher F0/energy plus normalized teacher block outputs. No vocoder or audio loss.
- Validate on held-out utterances, then rerun the 15-item gate.
- Continue only if it beats both 1.964 dB and 4.53 Hz. If it fails, try option 3 once, and do not repeat
  ordinary fine-tuning.

**Method** (`scripts/s016_refit_e2e.py`):
- Starts from the attempt-1 init.pt and uses Adam on every PitchStudent weight.
- Loss: `pitch_loss` (|ΔF0|/100 + |Δenergy|) plus the mean over 7 teacher blocks (shared LSTM, F0.0–2, N.0–2)
  of MSE on the kept channels ÷ that target's held-out variance, weight 1.
- Steps: 600 of 8 fresh train-split utterances each, alternating teacher and S014 durations. Rng 505.
- lr 2e-4, 20-step warmup, cosine decay to 0, grad clip 1.0.
- Validation every 100 steps on the 16 held-out utterances from attempt 1, which are excluded from the
  training pool. The checkpoint with the lowest held-out pitch loss is kept.
- The vocoder is copied unchanged. 732 s on the Mac CPU.

    python scripts/s016_refit_e2e.py --name s016-e2e --init artifacts/s015/runs/s016-prune/init.pt
    python scripts/s015_compare_s014.py s016-e2e-best --arm s015-full --checkpoint artifacts/s015/runs/s016-e2e/best.pt

| step | held F0 MAE (Hz) | held energy MAE | held pitch loss | held feature loss |
|---|---|---|---|---|
| 0 | 6.807 | 0.2056 | 0.2737 | 0.0084 |
| 100 | 6.061 | 0.1739 | 0.2345 | 0.0100 |
| 200 | 5.768 | 0.1555 | 0.2132 | 0.0100 |
| 300 | 5.394 | 0.1353 | 0.1892 | 0.0096 |
| 400 | 4.761 | 0.1308 | 0.1784 | 0.0095 |
| 500 | 4.518 | 0.1227 | 0.1679 | 0.0093 |
| 600 (best) | 4.469 | 0.1197 | 0.1644 | 0.0092 |

Reference (v6 best) on the same held-out set: 3.819 Hz / 0.099.
**15-item gate (best = step 600): 2.016 dB, F0 MAE 5.27 Hz.** Baseline is 1.964 dB / 4.53 Hz. **Failed.**
Not converted and not trained further. Moved on to option 3, done once.

Files: `artifacts/s015/runs/s016-e2e/` (best.pt e53c4e2b…e4696f00, last.pt f70bf359…8c7006bb, log.jsonl).
Log: `artifacts/s015/s016-e2e.log`. Script SHA-256 93c90886…d37d74d0.

## S016 attempt 3 (option 3): width moved from the energy branch to the F0 branch (2026-10-18)

Block 0's width must stay 2×lstm = 256 to keep Kokoro's identity shortcut, so only the block 1–2 width
(`half`) can move. `PitchStudent` gained `n_half` (the energy branch's block 1–2 width). Compiled int8 sizes from
`s015_size_probe.py pitch --pitch-config ...`:

| F0 half / N half | parameters | compiled bytes |
|---|---|---|
| 128 / 128 (attempts 1–2) | 2,735,362 | 2,872,329 |
| 192 / 64 | 2,809,090 | 2,949,720 |
| **192 / 80 (chosen)** | 2,858,770 | 3,000,919 |
| 176 / 80 | 2,776,834 | 2,917,715 |

Chosen: 192 / 80. It is 17,956 bytes under the 3,018,875 cap, and the projected app is 19,982,044 bytes.
It is run through the same pipeline: prune + layer-wise refit, then the end-to-end refit with the
attempt-2 settings, then the 15-item gate.

    python scripts/s016_prune_pitch.py --name s016-prune-f192n80 --vocoder artifacts/s015/runs/v4-prune/step10000.pt \
        --reference artifacts/s015/runs/v6-pitch-only/best.pt --config '{"width":256,"lstm":128,"inp":false,"half":192,"n_half":80}'
    python scripts/s016_refit_e2e.py --name s016-e2e-f192n80 --init artifacts/s015/runs/s016-prune-f192n80/init.pt
    python scripts/s015_compare_s014.py s016-e2e-f192n80-best --arm s015-full --checkpoint artifacts/s015/runs/s016-e2e-f192n80/best.pt

The user confirmed 192 / 80 (block 0 stays 256 on both branches), asked for held-out energy error to be recorded
against the v6 reference of 0.099, and said to stop if either gate number is missed.

Prune + layer-wise refit:
- Kept: L 256/512; per branch I1 256/512; F0 I2/O2/I3 192/256; N I2/O2/I3 80/256.
- Refit R²: F0 branch ≥0.904 (weakest: block 2 conv2 0.904, block 1 conv1x1 0.927). N branch ≥0.963.
- Held-out: sliced only 41.27 Hz / 4.652; refit 6.221 Hz / 0.2351. 699 s.

End-to-end refit (same 600-step settings, 711 s):

| step | held F0 MAE (Hz) | held energy MAE | held pitch loss | held feature loss |
|---|---|---|---|---|
| 0 | 6.221 | 0.2351 | 0.2973 | 0.0088 |
| 100 | 5.449 | 0.1748 | 0.2293 | 0.0107 |
| 200 | 4.871 | 0.1573 | 0.2060 | 0.0105 |
| 300 | 4.904 | 0.1459 | 0.1950 | 0.0104 |
| 400 | 4.493 | 0.1322 | 0.1771 | 0.0102 |
| 500 | 4.342 | 0.1296 | 0.1730 | 0.0100 |
| 600 (best) | 4.278 | 0.1265 | 0.1693 | 0.0100 |

Reference (v6 best) on the same held-out set: 3.819 Hz / **0.099 energy**.

| S016 candidate | held F0 (Hz) | held energy | 15-item dB | 15-item F0 (Hz) |
|---|---|---|---|---|
| attempt 1, 128/128 layer-wise refit | 6.807 | 0.2056 | 2.235 | 8.39 |
| attempt 2, 128/128 + end-to-end refit | 4.469 | 0.1197 | 2.016 | 5.27 |
| attempt 3, 192/80 + end-to-end refit | 4.278 | 0.1265 | 2.007 | 5.00 |
| baseline v6 best (S015 pitch) | 3.819 | 0.099 | 1.964 | 4.53 |

**Result: S016 failed; stopped.** Attempt 3 (2.007 dB / 5.00 Hz) misses both gate numbers. Moving width to F0
improved F0 (5.27 → 5.00 Hz on the 15 items, 4.47 → 4.28 Hz held-out). The thinner energy branch made held-out
energy worse (0.120 → 0.127, against the 0.099 reference). No Core ML conversion, no long training, no build, no
listening. README unchanged. This was the one width change. **S014 mixed8 stays the accepted result.**

Files: `artifacts/s015/runs/s016-prune-f192n80/` (init.pt a0f50a6e…80b1dcba, prune.json 6c9796db…8a2cf3c1),
`artifacts/s015/runs/s016-e2e-f192n80/` (best.pt cc4772db…39d30323, last.pt 516fda08…e60f29b4, log.jsonl).
Logs: `artifacts/s015/s016-prune-f192n80.log`, `s016-e2e-f192n80.log` and `s016-e2e-f192n80-compare.log`.
Final script SHA-256: s016_prune_pitch.py 529d8976…7bc7f78c, s015_downstream.py 33d95021…e06499d5.
Size-probe outputs: `size_pitch_int8_*_half192_n_half{64,80}.json` and `..._half176_n_half80.json`.

VM: `<gpu-vm>` is still running and idle, as the user asked; S016 did not use it.

## S017: compiled Core ML models under 25,000,000 bytes (2026-10-18)

Chain: S014 width-256 frontend (`kokoro_student_w256`, 6,076,253 compiled bytes) → full Kokoro flexible int8 pitch
(`kokoro_pitch_int8_flex`, 8,622,252) → v4-prune step-10000 vocoder int8 (`kokoro_s015_vocoder_int8`, 9,569,102).
The vocoder runs on `.cpuAndGPU`, like the S014 generator; the other two run on `.all`. Swift does the duration
sigmoid-sum, SourceDSP and iSTFT, copied from S014. Output is 600 samples per frame at 24 kHz.

Scope correction (user, 2026-10-18): the deliverable is the set of smaller Kokoro Core ML models, not a production
app. The iOS target is only a validation harness. Its optional phoneme field and listening UI are conveniences, not
requirements or part of the model-size result. No Misaki, eSpeak or lexicon is bundled.

New sources: `benchmark/ios/KokoroSize-S017/` (CoreMLChain, Phonemes + VoiceTable, Bench, ContentView, App,
HiddenListenView under `#if S017_LISTEN`), `benchmark/ios/Resources-S017/` (symlinks) and `benchmark/ios/project-s017.yml`
→ `KokoroSizeS017.xcodeproj`. S014's `KokoroSize/`, project and resources are untouched.

Configurations (user chose a separate Listen config): Release and Listen both use `DEPLOYMENT_POSTPROCESSING`,
`STRIP_INSTALLED_PRODUCT`, `STRIP_STYLE all` and `DEAD_CODE_STRIPPING`. Listen adds `S017_LISTEN` (the hidden
listening view and `-make-listen`). Models, chain, DSP, voice table and settings are the same in both.

    cd benchmark/ios && xcodegen generate --spec project-s017.yml
    xcodebuild -project KokoroSizeS017.xcodeproj -scheme KokoroSize -configuration {Release,Listen} \
        -destination generic/platform=iOS -derivedDataPath build/s017-{Release,Listen} CODE_SIGNING_ALLOWED=NO build
    xcodebuild ... -derivedDataPath build/s017-{c}-device CODE_SIGN_STYLE=Automatic DEVELOPMENT_TEAM=<team-id> \
        CODE_SIGN_IDENTITY="Apple Development" PROVISIONING_PROFILE_SPECIFIER= -allowProvisioningUpdates build

Size (logical bytes, `stat`, unsigned .app):

| item | bytes |
|---|---|
| student + pitch + vocoder (compiled) | 24,267,607 |
| af_heart.bin (fp32, 510×256) | 522,240 |
| KokoroSize binary (stripped) | 181,936 |
| bench.json (slim: ids + sample counts) | 3,367 |
| Info.plist + PkgInfo | 930 |
| **production Release total** | **24,976,080** (23,920 under) |
| Listen total (not graded) | 25,091,384 |

The first production build carried the full 29,768-byte S014 bench.json (style rows, text) and came to 25,002,537,
2,537 bytes over. `scripts/s017_bench_inputs.py` writes the slim file (style rows come from af_heart.bin; the script
asserts the stored S014 rows equal the voice-table rows exactly). That alone met the target. **No fp16 voice table and
no energy-branch cut were needed**, so neither was tried.

Signed device builds, for information only: Release 25,014,678, Listen 25,130,214. Signing adds 38,598 bytes
(embedded.mobileprovision 12,311, `_CodeSignature/CodeResources` 6,735, and 19,552 of code signature in the binary).
The graded number is the unsigned one, as for S014.

Build identity (Release vs Listen): source package hashes are student ade84baa…f644e63c, pitch 13d59179…f920c439f and
vocoder c86ce4fd…bc2303f. The compiled hashes, excluding `coremldata.bin`, are identical in both builds: student
1efa3b6d…27419, pitch 3fc5c3cf…6f0fc, vocoder 4a404143…f04d3. `coremldata.bin` (554 bytes) differs, but it also
differs between two compiles of the same package (metadata order), so it is compiler non-determinism. bench.json
(976e3875…7d589) and af_heart.bin (d583ccff…4a8f0b) match. Waveform-hash check (`-hash`, 3s case) still to run on device.

15-item gate (`scripts/s015_compare_s014.py`, new arm `s017ml-full` = Core ML S014 int8 pitch + Core ML vocoder;
backup of the previous script `artifacts/s017/s015_compare_s014.py.before`, now 18d430cd…4bcfc8):

    python scripts/s015_compare_s014.py s017 --arm s017ml-full --coreml artifacts/s015/coreml/v4-prune-step10000 \
        --checkpoint artifacts/s015/runs/v4-prune/step10000.pt
    python scripts/s015_compare_s014.py s017 --arm s014m8-full
    python scripts/s015_compare_s014.py s017 --summary

| arm | DTW dB | RMS Δ dB | 6–8k | 8–10k | 10–12k | F0 MAE Hz | finite |
|---|---|---|---|---|---|---|---|
| **s017ml-full** | **1.747** | −0.17 | +0.17 | +0.21 | +0.33 | 0.13 | all |
| s014m8-full (reproduced) | 1.862 | −0.07 | +0.45 | −0.19 | +0.70 | 0.13 | all |

S017 per-item DTW 1.579–1.875. Energy MAE vs fp32 Kokoro 0.003–0.004. DTW vs the fp32 F0/N + fp32 v4 chain 0.44–0.92.
Waveform SNR against that chain is −3.7 to +6.0 dB: int8 F0 drifts the sine phase over long clips, so SNR is not a
useful quality measure here. Sample counts are correct on every item. **The ≤1.86 dB gate passes.**

Still to run: iPhone 12 speed gate on production Release (`-coreml-speed`, median RTF ≤0.30 for 3/7/15/30 s), the
`-hash` waveform match between Release and Listen, and blind listening on Listen with the user. The phone showed
`unavailable` in `devicectl`. The same bundle id replaces the S014 app, so back up its Documents first. README and
GOAL are unchanged until those pass. **S014 mixed8 stays the accepted result until then.**

Files: `artifacts/s017/` (af_heart_f32.bin, bench-s017.json, build logs), `artifacts/s015/compare/s017/`,
`artifacts/s015/s017-compare.log`. Script SHA-256: s017_bench_inputs.py ede90a9d…4d35bad7.

VM: on the user's instruction (2026-10-18), `<gpu-vm>` (<gpu-machine-type>, <gcp-zone>,
project <gcp-project>) was stopped, not deleted. Status `TERMINATED`; its disk is kept. The remaining S017 steps do not use it.

### S017 on the iPhone 12 (2026-10-18)

Before installing, the existing app's Documents (S014 results and listening files, 496 files, 100 MB) were copied to
`artifacts/s017/device-backup-before-s017/Documents`. S017 uses the same bundle id.

**The vocoder on `.cpuAndGPU` is killed by iOS at long inputs.** With the first production build, the speed test
passed 3 s (median RTF 0.218, then 0.208 on a rerun) and 7 s (0.149, then 0.140), then stopped at 15 s both times.
The first run's process stayed alive but stalled for 17+ minutes; the second run's process disappeared. iOS wrote
no KokoroSize crash report and no JetsamEvent. A diagnostic build (`S017_DIAG`, separate derived data
`build/s017-diag-device`; it logged each `prediction` call to Documents/s017-stage.txt) showed the student and pitch
models finishing at 567 frames (4 ms, 26 ms). The vocoder call at text [1,512,567] / spectrum [1,22,68041] never
returned and the process was killed. At 272 frames the same call took 0.86 s. Logs: `artifacts/s017/s017-stage*.txt`.
This vocoder had never run on the phone before; its S015 latency was measured on the Mac only.

Diagnostic build with the vocoder on `.cpuOnly` (hot phone, thermal `serious`; not graded):
3 s 0.076, 7 s 0.085, 15 s 0.105, 30 s 0.117 median RTF (`artifacts/s017/coreml-speed-diag-voc-cpu.json`).

Decision (user): the production vocoder loads on `.cpuOnly`; the frontend and pitch stay on `.all`. CPU+GPU cannot
cover the 15 s and 30 s cases. CPU-only is also what the 1.747 dB Mac gate ran (`s015_compare_s014.py` loads
Core ML with `CPU_ONLY`), so the gated audio and the phone audio use the same compute path. The diagnostic code was
removed. The same rebuild added a caption under the phoneme field ("Kokoro phonemes (IPA). Example = "hello
world".") after the user found the phoneme default text confusing (user chose the caption option).

Rebuilt sizes (unsigned): **production Release 24,976,568** (binary 182,424; +488 for the caption; 23,432 under),
Listen 25,091,384. Signed: Release 25,015,174, Listen 25,130,214.

Waveform hash, 3 s case, on the phone (`-hash`): Release and Listen both give 85,800 samples, SHA-256
04be46d5b21be4c9faa464f38623c7f70daaedca00f560e568c50cbb1813fe61. **Identical.** (143 frames: the S014
frontend's durations; bench.json's 87,600 is the original Kokoro length. S014 also made 143 frames on the phone.)

**Graded speed run: production Release, vocoder `.cpuOnly` (`-coreml-speed`, 2 warmups + 10 timed runs).**
The phone was charging over USB and did not cool; the user chose to run as is. Thermal state was `serious` on every
row, so these numbers are pessimistic. Results: `artifacts/s017/coreml-speed-s017.json` (e2e8afbd…).

| case | audio s | median RTF | p90 | max | sample count | pass ≤0.30 |
|---|---|---|---|---|---|---|
| 3 s | 3.58 | 0.060 | 0.062 | 0.063 | ok | yes |
| 7 s | 6.80 | 0.072 | 0.074 | 0.075 | ok | yes |
| 15 s | 14.18 | 0.096 | 0.101 | 0.102 | ok | yes |
| 30 s | 31.25 | 0.109 | 0.111 | 0.112 | ok | yes |

**Speed gate passes.** For reference, S014 mixed8 ran 0.17–0.18 on this phone. Remaining: blind listening on the
Listen build with the user. README and GOAL stay unchanged until it passes.

Listening prep (Listen build): the S017 HiddenListenView copy still had `firstEntry = 24`, which would replay entries
the user heard in the S014 sessions (0–31). It now starts at 32. Rebuilt and installed Listen. Pushed
`artifacts/s014/listen/sentences.json` and `ref/` (120 original-Kokoro wavs; identical to the phone's listen-s014
copy; every style row equals the af_heart voice-table row, max diff 0.0) to Documents/listen-s017. `-make-listen`
made 120 S017 clips on the phone in about 75 s: 0 non-finite, samples = 600 × frames for all, longest 553 frames,
max peak 0.658, no clipping. Per-clip checks: `artifacts/s017/listen-frames.json`. Blind session: 8 pairs vs
original Kokoro, entries 32–39 of the fixed shuffled order.

#### S017 blind listening, session 1 (listener index 0, entries 32-39)

8 pairs (h091 h060 h011 h041 h077 h117 h118 h104): all 8 tie, MOS 5/5 on both sides. Not counted as a
pass. Votes came about 6-7 s apart, but pairs 1-7 hold about 55 s of audio (h091 is 12.3 s), so not every
clip was played to the end. Votes kept in artifacts/s017/listen-s017.json (same bytes as the phone copy
pulled before the next install, listen-s017-session1-prephone.json).

#### S017 blind listening, session 2 setup (2026-10-18)

The user's rules: 8 new unheard sentences, each clip about 2-4 s and none over 5 s. A/B sides stay random,
and a vote is possible only after both clips play to the end. Listen-only changes (HiddenListenView.swift,
already left out of the production build):
- Listener k >= 1 takes the k-th group of 8 entries from shuffled-order entry 40 on whose reference and S017
  wavs both last 2.0-4.0 s. Selection uses wav lengths only, never the text. 27 entries qualify. Listener 1
  gets h062 h073 h088 h028 h105 h081 h053 h084, lengths 2.67-3.85 s. Sides are seeded by 0xA5_0000 + 1.
- ListenPlayer (a copy of AudioPlayer with a `.dataPlayedBack` completion) marks a side as heard only if the
  same play is still current and at least the clip length minus 0.1 s has passed. Prefer/Tie and MOS stay
  disabled until A and B have both been heard. Next also needs both.
- Listen sizes after the change: unsigned 25,092,368, signed 25,131,190. Production Release is not rebuilt
  because no shared file changed.

#### S017 blind listening, session 2 result (2026-10-18)

Listener "listener-1" (index 1), 8 pairs, play-to-end gate on: all 8 tie, MOS 5/5 for both original and S017
(means 5.00 / 5.00). Original on side A in 4 of the 8 pairs (h088 h028 h105 h053). Votes came 8-46 s apart,
longer than each pair's A+B audio (5.5-7.6 s). Session-1 rows are unchanged in the file. The pulled file is
artifacts/s017/listen-s017-after-session2.json (16 rows).

#### S017 grade (2026-10-18)

- **Size:** 24,267,607 compiled-model bytes, measured in the Release `.app`: kokoro_student_w256 6,076,253,
  kokoro_pitch_int8_flex 8,622,252, kokoro_s015_vocoder_int8 9,569,102. With af_heart.bin (522,240):
  24,789,847. The harness `.app` (24,976,568 unsigned) is recorded, not graded.
- **Speed:** median RTF 0.060 / 0.072 / 0.096 / 0.109 for 3 / 7 / 15 / 30 s. PASS.
- **Sound:** short blind session (listener index 1): 8 ties, 0 reference wins, 0 S017 wins, with the
  play-to-end gate on. No metallic edge reported. MOS (recorded, not graded): 5/5 on both sides for every pair.
  Session 1 (8 ties) was not counted because not every clip was played to the end. PASS by the GOAL.md rule.
- **Tier:** S017 passes speed, sound and size: **Ultra (<= 25 MB)**. It replaces S014 mixed8 as the
  accepted result. The 60-minute loop was not run (not a gate).


### S017 validation gaps: full-chain eval, output check, four-length prep (2026-10-19)

**1. The old gate arm did not run the student as Core ML.** `s017ml-full` in `scripts/s015_compare_s014.py` runs the
S014 student in PyTorch fp32 (`student(...)`), takes durations from `teacher.frames_from_logits` (torch.round,
half-to-even), uses the torch sine source (`sine_spectrum`, torch seed) and torch `stft.inverse`. Only the pitch and
vocoder were Core ML. Its 1.747 dB stays as recorded; no existing arm or result was changed.

New `scripts/s017_eval_full_chain.py` runs the shipped chain with every Core ML stage on `CPU_ONLY`:
`kokoro_student_w256` -> Swift duration rule ported from `CoreMLChain.durations` (Float sigmoid, sequential Float
sum over 50 bins, round half away from zero, min 1) -> `kokoro_pitch_int8_flex` -> Swift `SourceDSP` ported to numpy
(SplitMix64 noise, 9 harmonics, merge, tanh, 20/5 STFT) -> `kokoro_s015_vocoder_int8` -> Swift `inverseSTFT` ported.
Same 15 items, same reference (fp32 S014 student + fp32 Kokoro, torch sine seed) and same metric (`dtw_logmel_db`)
as the gate. Nothing tuned.

    .venv-convert/bin/python scripts/s017_eval_full_chain.py
    .venv-convert/bin/python scripts/s017_eval_full_chain.py --frames

**Result: mean 1.787 dB (gate <= 1.86, pass), not 1.747.** RMS -0.17 dB, 6-8k +0.16, 8-10k +0.21, 10-12k +0.32,
all finite, samples = 600 x frames on every item. The Mac, CPU_ONLY run differs from the phone's .all compute units
in the student and pitch fp16 paths, so this is the Mac figure for the shipped chain, not a phone measurement.

| item | frames (Core ML student) | frames (old PyTorch student) | samples | DTW dB |
|---|---|---|---|---|
| 3s | 143 | 143 | 85,800 | 1.678 |
| 7s | 272 | 272 | 163,200 | 1.687 |
| 15s | 567 | 568 | 340,200 | 1.918 |
| 30s-0 | 606 | 605 | 363,600 | 1.987 |
| 30s-1 | 644 | 644 | 386,400 | 1.823 |
| val0-val9 | 207 210 240 698 209 269 606 279 693 116 | same except val8 (692) | 69,600-418,800 | 1.707 1.644 1.698 1.811 1.712 1.841 1.856 1.694 2.168 1.588 |

(val rows in order val0..val9.) The three
items with a different frame count (15s, 30s-0, val8; the Core ML fp16 student rounds one token's duration the other
way) are the three highest, since the fp32 reference keeps the PyTorch student's length and differs by 1 frame
(600 samples) there. That is the real cost of the shipped student; it was not adjusted.

Frame-count proof (Swift rounding): the port matches the phone. (a) Speed-run audio from the phone
(`artifacts/s017/coreml-speed-s017.json`): 3s 143 frames, 7s 272, 15s 567, 30s 1,250 (=606+644), equal to the Mac
Core ML counts. (b) All 120 listen clips: Mac frame counts equal the phone's `artifacts/s017/listen-frames.json`
for 120 of 120 (0 mismatches). No half-way ties occurred (`duration_half_ties` 0 on all items), so the half-away
vs half-even difference did not matter here, but the Swift rule is the one ported. The PyTorch student differs from
the phone on 3 of 15 items, the Core ML student on 0.

Saved: `artifacts/s017/eval-full-chain/` (`eval.json` with per-item metrics, frames, samples, per-model package
SHA-256, compiled logical bytes, and output wav SHA-256s; 15 wavs), summary `benchmark/results/size-s017-eval.json`
(no paths). Compiled logical bytes: student 6,076,253, pitch 8,622,252, vocoder 9,569,102, total 24,267,607. Package
SHA-256: student e9bd7b64..., pitch 1025f208..., vocoder 4c004ca4... (hash over the .mlpackage files, sorted).

**2. `sampleCountMatched` replaced.** `Bench.swift` hardcoded `true`. bench.json's `samples` is the original Kokoro
length (3s: 87,600), not the student's frame count (85,800), so it cannot be the expected count. The field is now
`outputShapeValid`: true only if every timed run produced audio, a whole number of 600-sample frames, with the same
length on all 10 runs of a chunk. Computed outside the timed window; audio code unchanged. The key in
`coreml-speed.json` changes, so old results (`artifacts/s017/coreml-speed-s017.json`) keep the old name. Rebuilt
unsigned: **Release .app 24,976,568 bytes** (same as before; binary 182,424), under 25,000,000. `-hash` was **not
run**: the iPhone showed `unavailable` in `devicectl`. The 3 s waveform is unchanged by construction (no synthesis
code touched); the device check (expected 04be46d5...fe61, 85,800 samples) is still to run.

**3. Four-length blind test, prepared (run later the same day; result in "S017 four-length blind test (2026-10-19)" below).** In `HiddenListenView.swift` (all under `#if S017_LISTEN`): a
segmented Picker above "Listener name" with "Short hidden" (default, unchanged behavior) and "Four lengths"; the
existing Start session button starts the selected set. Four lengths = the frozen 3s, 7s, 15s and 30s cases in that
order, 2 pairs with the reference on A and 2 on B (seeded by `0xF0_0000 + listener index`), the same play-to-end gate,
MOS and tie/prefer controls. `ListenResult` got an optional `set` field ("short" or "four"); short-session rows
already in `listen-s017.json` (16) decode unchanged. Four-length votes go to a separate file,
`Documents/listen-s017-four.json`, with its own listener index. "Make S017 clips" now also writes the four S017 clips
(same CoreMLChain, bundled bench.json and af_heart rows, 30s = 30s-0 then 30s-1 joined, same finite/peak/rms/frames
checks) to `listen-out-s017/four/s017/{3s,7s,15s,30s}.wav` with `frames.json`.

Reference source: the same as the earlier four-length tests (S004, S007, S011 played the bundled
`benchmark/inputs/{3s,7s,15s,30s-0,30s-1}.wav`, original Kokoro `af_heart` from kokoro-onnx 0.6.1 with eSpeak
phonemes; the 30s case played as its two wavs in order). `scripts/s017_four_length_prep.py` copies these to
`artifacts/s017/four-length/ref/{3s,7s,15s,30s}.wav` (30s joined without a gap; 87,600 / 165,000 / 340,200 / 748,200
samples). The reference has slightly different lengths from S017 (S017 3s is 85,800); that was also true before.

Push (run only when the phone is connected; back up `Documents/listen-s017.json` first), from the project root:

    D=<device-id>; B=com.kokorobench.KokoroBench
    xcrun devicectl device copy from --device $D --domain-type appDataContainer --domain-identifier $B \
        --source Documents/listen-s017.json --destination artifacts/s017/listen-s017-before-four.json
    xcrun devicectl device install app --device $D benchmark/ios/build/s017-Listen-device/Build/Products/Listen-iphoneos/KokoroSize.app
    for c in 3s 7s 15s 30s; do xcrun devicectl device copy to --device $D --domain-type appDataContainer \
        --domain-identifier $B --source artifacts/s017/four-length/ref/$c.wav \
        --destination Documents/listen-s017/four/ref/$c.wav; done
    xcrun devicectl device process launch --device $D --terminate-existing $B -- -make-listen

Listen builds: unsigned **25,110,928** bytes (binary 316,784), signed 25,149,782. Release is 24,976,568 (above).

Still pending: the four-length blind run by the listener. The complete documented sound gate (four lengths plus unseen
sentences) is not yet shown as passed; the short blind sessions (8 ties, 0 reference wins) stand as recorded.

#### S017 phone checks after the validation fixes (2026-10-19)

- **Release `-hash`** (signed Release with the `outputShapeValid` change): 3s 85,800 samples, sha256
  04be46d5b21be4c9faa464f38623c7f70daaedca00f560e568c50cbb1813fe61, identical to the earlier Release and Listen
  builds. The file was written by this run (phone timestamp 13:47). Copy: artifacts/s017/hash-S017-after-bench-fix.json.
- **Votes backup** before the installs: artifacts/s017/listen-s017-before-four.json, same bytes as the post-session-2 file.
- **Four-length prep on the phone:** the signed Listen build was installed, the four reference wavs were pushed to
  Documents/listen-s017/four/ref/, and `-make-listen` made the four S017 clips. Checks
  (artifacts/s017/four-length/phone-frames.json): 3s 143 frames / 85,800 samples, 7s 272 / 163,200, 15s 567 /
  340,200, 30s 1,250 / 750,000. 0 non-finite, peak 0.38-0.47, RMS -26.3 to -25.7 dB. The frame counts equal the Mac
  full-chain eval. The listening run itself has not happened.

#### S017 four-length blind test (2026-10-19)

One listener, the Listen build's "Four lengths" set. These are the frozen 3s, 7s, 15s and 30s cases against the original
Kokoro `af_heart` reference wavs (benchmark/inputs), with randomized sides (reference on B for 3s and 7s, on A for 15s
and 30s) and the play-to-end gate on. Results: 3s tie, 7s tie, 15s tie, 30s S017 preferred. That is **0 reference wins**,
3 ties and 1 S017 win. MOS (recorded, not graded): 5/5 on both sides for every pair. Gaps between votes were 17, 39 and
69 s, each longer than that pair's A+B audio (about 14, 28 and 62 s). The short-session rows in listen-s017.json are
byte-identical to the file after session 2. Votes: artifacts/s017/listen-s017-four.json.

**Complete documented sound gate (four lengths plus unseen sentences): PASS.** It covers the four-length test above and
the short hidden session (8 ties, 0 reference wins). With speed, the 1.787 dB full-chain objective check and 24,267,607
compiled-model bytes, S017 stands as the accepted Ultra-tier result.

The user confirmed that the same person ran all three S017 sessions (the two short sessions and the four-length one).
The one-listener gate in GOAL.md therefore holds as written. `listenerIndex` numbers the sessions, not people.

## Multi-tier release staged (2026-10-19)

The Hugging Face release candidate (`release/huggingface`, local only, nothing uploaded) holds every size tier that passed speed and sound, one self-contained folder each. `release/github` carries the harness, project spec and resources link folder for each tier. Compiled-model bytes are the logical bytes of the `xcrun coremlcompiler` output, measured fresh for this release and equal to the `.mlmodelc` sizes inside the unsigned Release builds:

| Tier folder | Experiment | Compiled models (bytes) | Shapes | Unsigned `.app` (bytes, not graded) |
| --- | --- | ---: | --- | ---: |
| `500mb/` | S002 shared fp32 | 327,680,270 | fixed (five frozen shapes) | 330,966,116 |
| `200mb/` | S007 int8 decoder and generator | 168,492,559 | fixed (five frozen shapes) | 171,778,405 |
| `100mb/` | S011 int8/int4 stack | 96,203,053 | fixed (five frozen shapes) | 99,490,915 |
| `50mb/` | S014 mixed8 | 42,582,199 | flexible | 46,109,125 |
| `25mb/` | S017 (recommended) | 24,267,607 | flexible | 24,976,568 |

The 200mb total (168,492,559 = text 25,469,752 + prosody 53,708,437 + pitch 33,959,547 + int8 decoder 34,151,789 + int8 generator 21,203,034) was not recorded before; it is new here.

Left out as failed or unscored: S001, S006, S008, S009, S010 sweep packages, S012, S013, the S014 int4 pilot, S015, S016. Only the 25mb tier ships a voice table (`af_heart_f32.bin`); the other tiers take their `af_heart` rows from the frozen `benchmark/inputs/bench.json`. The staged harness sources differ from the working ones only by `#if` branches: a new `S002` flag (shared fp32 decoder and generator, same code path as the S003 run) and the S012 branch removed. The working sources are unchanged; the stage script re-applies the patch on every run (`patch_once`, exact-match asserts). All five tiers built unsigned (`CODE_SIGNING_ALLOWED=NO`, generic iOS destination) from the staged tree with the HF tree linked as `artifacts/hf/`. No device, listening or speed run was done for this step.

## S017 four-length blind test, second listener (2026-10-19)

A second person (listener-2) ran the Listen build's "Four lengths" set on the iPhone 12. They did not run the short hidden-sentence set. Votes were pulled from the app's Documents/listen-s017-four.json into `artifacts/s017/pull-1511/` (8 rows: listener-1's earlier 4 plus these 4). `artifacts/s017/listen-s017-four.json` now holds all 8 rows, and the earlier 4-row copy is kept as `listen-s017-four-listener1.json`.

| Length | Preferred | Scores (S017 / reference) |
|---|---|---|
| 3s | S017 | 5 / 4 |
| 7s | S017 | 5 / 5 |
| 15s | reference | 4 / 5 |
| 30s | S017 | 5 / 4 |

Result: 3 S017 preferences, 1 reference preference. Under the written GOAL rule ("passes unless the listener prefers the reference"), this session has one reference preference. The owner judges it acceptable, since the smaller model was preferred in 3 of 4 pairs, and S017 stays the accepted result. Listener-1's counted results are unchanged: hidden session 2 (8 ties) plus four lengths (3 ties, 1 S017 win), 12 pairs, 0 reference wins; hidden session 1 (no play-to-end gate) stays not counted. Across both listeners: 16 pairs, 11 ties, 4 S017, 1 reference. Listener-1 is the project author; the test was blind but not independent, and the author judged listener-2's single reference preference acceptable.


## S018: retrained student (2026-10-21)

This entry is a summary written for publication. S018 replaced the student in the `25mb/` and `50mb/` tiers.

The `25mb/` and `50mb/` tiers now ship a retrained `kokoro_student_w256.mlpackage` (the S018 student), published as trained. It replaces the S017 student in both tiers; nothing else in those tiers changed. The previous version stays available: Hugging Face commit [`e7ed75f144376a2d11377cb8af706c3c54b0b0f6`](https://huggingface.co/<handle>/kokoro-82m-coreml-size-tiers/tree/e7ed75f144376a2d11377cb8af706c3c54b0b0f6) and the GitHub history.

**The problem.** The S017 student mispronounced common words when the output was transcribed by speech recognition: for example chapter, catch, nice, okay, gave, life and courage came out as other words ("chapter" as "adapter", "catch" as "count", "nice" as "ness"). Blind listening and the log-mel distance had not caught it.

**The cause.** The sounds involved (the "ch" and "j" sounds, long vowels such as the "ay" in "gave" and the "i" in "nice", the "w" sound) are rare in running text, and the student's training averaged its loss over frames, so a sound that lasts a few frames could fail at almost no cost.

**The fix.** The student was fine-tuned from the previous student's checkpoint on general text plus targeted sentences, with a higher weight on those sounds in per-token losses and an added log-mel term. The package has the same interface and the same compiled size (6,076,256 bytes for the student; the 25mb tier totals 24,267,610 bytes). The recipe is generic and is published: `docs/README_RETRAIN.md` with the scripts `scripts/s018_*.py`.

**Results** (`af_heart`; whisper.cpp `ggml-medium.en` transcribes each render and the word error rate is computed against the input text, lowercased and without punctuation; same token ids and style rows for every column; Mac Core ML, CPU only; 25mb chain):

| Check | Reference Kokoro | S017 student | S018 student | Source |
| --- | ---: | ---: | ---: | --- |
| Held-out WER (751 words, public-domain book text not used in training) | 2.93% | 19.57% | **3.86%** | `benchmark/results/size-s018-score-heldout-25mb.json` |
| Neutral A/B WER (135 words, everyday sentences) | 1.48% | 20.00% | **0.74%** | `benchmark/results/size-s018-score-abn-25mb.json` |
| DTW log-mel against fp32 Kokoro, 15 items (dB, lower is better) | n/a | 1.787 | 2.576 (worse) | `benchmark/results/size-s018-dtw-25mb.json` |

- The WER gate (within 1 percentage point of the reference on the same set and voice, and no word the reference says correctly failing in more than one sentence) is met on both sets: held-out +0.93 points, neutral A/B -0.74 points, no repeated-word failures.
- **The DTW log-mel distance is worse than S017's** (2.576 dB against 1.787 dB, +0.789 dB; the bar S015 to S017 used was 1.86 dB). It is recorded, not a gate. The audio differs more from the reference's log-mel while the words are right.
- 50mb chain (same student with the S014 int4 decoder and mixed8 generator): held-out 2.93% / 19.57% / **3.86%** (reference / S017 / S018; `benchmark/results/size-s018-score-heldout-50mb.json`), neutral A/B 1.48% / 22.22% / **0.74%** (`benchmark/results/size-s018-score-abn-50mb.json`), DTW 1.901 dB (S017) against 2.754 dB (S018, worse; `benchmark/results/size-s018-dtw-50mb.json`).
- The 100, 200 and 500 MB tiers do not use the student and are unchanged.

**Speed and listening for the S018 student:**

- 25mb: Median RTF (iPhone 12): 3s 0.062, 7s 0.073, 15s 0.092, 30s 0.103 (gate 0.30; source `benchmark/results/size-s018-speed-25mb.json`). Sound: 12 ties of 12 blind pairs, 0 reference preferred (`benchmark/results/size-s018-25mb-listen-s018.json`, `benchmark/results/size-s018-25mb-listen-s018-four.json`).
- 50mb: Median RTF (iPhone 12): 3s 0.184, 7s 0.193, 15s 0.195, 30s 0.210 (gate 0.30; source `benchmark/results/size-s018-speed-50mb.json`). Sound: 12 ties of 12 blind pairs, 0 reference preferred (`benchmark/results/size-s018-50mb-listen-s018-50mb.json`, `benchmark/results/size-s018-50mb-listen-s018-50mb-four.json`).

Figures in this document that are labeled S017 (or "previous student") belong to the previous student, not to the S018 student.

Files: held-out set `benchmark/inputs/s018-heldout/`, neutral A/B set `benchmark/inputs/s018-ab-neutral/`, results `benchmark/results/size-s018-*.json`, recipe `docs/README_RETRAIN.md` and `scripts/s018_*.py`.
