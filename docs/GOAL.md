# Goal

Build useful offline Kokoro Core ML model bundles at progressively smaller size tiers while preserving speed and sound on a physical iPhone 12.


## Success

Speed and sound must pass. Size decides how strong the result is.

1. **Speed:** warm median RTF <= 0.30 for each 3, 7, 15, and 30 second test. Discard 2 warmups and take the median of 10 runs.
2. **Sound:** one listener, blind, against reference Kokoro with `af_heart`. Use the four lengths and a few unseen sentences. The build passes unless the listener prefers the reference. A metallic edge, wrong pronunciation, or broken rhythm is a fail. Ties count for neither.
3. **Size:** total logical bytes of every compiled Core ML model required by the synthesis chain. Report required voice data and the complete test-harness `.app` separately. Models downloaded later or omitted required model stages do not count.

The 60-minute loop is not a gate. A panel of listeners, MOS scores, and confidence intervals are not gates. Record them if they are run.

The size tiers are:

| Tier | Compiled Core ML models |
| --- | ---: |
| Baseline | <= 500 MB |
| Stretch | <= 200 MB |
| Aggressive | <= 100 MB |
| Research | <= 50 MB |
| Ultra | <= 25 MB |

The result is the smallest model tier that still passes speed and sound. A larger tier may pass when a smaller tier fails. Record every tier that was tested.

S018 update: the `25mb/` and `50mb/` tiers now carry a retrained student that fixes S017's mispronunciation of common words (held-out WER 3.86% against 19.57% for S017; reference Kokoro 2.93%; same interface, same compiled size, 24,267,610 bytes for 25mb). Its DTW log-mel distance is worse than S017's. On the iPhone 12 the new student passes speed (median RTF 0.062 to 0.103 for 25mb) and the blind listening rule (12 ties in 12 pairs per tier, one listener, the author); see the README. The speed, sound and DTW figures below describe the previous (S017) student and are kept as written.

The S017 bundle with the S018 student is the current best result: **24,267,610 compiled-model bytes**, under the 25 MB Ultra tier. The required fp32 `af_heart` voice table is a separate 522,240 bytes, so models plus voice are 24,789,850 bytes. The unsigned Release harness `.app` is 24,976,571 bytes (recorded, not graded). The three sizes are reported separately everywhere.

- **Speed (previous S017 student):** passes (median RTF 0.060 to 0.109 against the 0.30 gate) on one iPhone 12.
- **Sound (previous S017 student):** a short one-listener blind session on 8 unseen sentences (2 to 4 s clips, randomized sides, play-to-end gate) gave 8 ties and 0 reference wins. The documented four-length blind test (3, 7, 15, 30 s): 3 ties, 1 S017 preference (30 s), 0 reference wins. Both meet the rule above (the build passes unless the listener prefers the reference; ties count for neither), so the complete documented sound gate passes for that listener. A second listener (listener-2) ran the four-length test only: 3 S017 preferences and 1 reference preference (15 s). Under the strict written rule that is one reference preference; the author judges it acceptable because the smaller model was preferred in 3 of 4 pairs. Listener-1 is the project author; the test was blind but not independent, and the author judged listener-2's single reference preference acceptable.
- **Objective check (recorded, not a gate):** the full shipped chain scores 1.787 dB mean log-mel distance against the 1.86 dB threshold on the 15-item set (PASS); this is a Mac Core ML CPU_ONLY figure.

The S017 bundle is the S014 width-256 student frontend, Kokoro int8 flexible-length pitch, and the S015 pruned int8 student vocoder. On the iPhone 12 the vocoder runs CPU-only. The 60-minute loop was not run.

The previous result, S014 mixed8, is 42,582,199 compiled-model bytes (50 MB Research tier, 12 ties, 0 reference wins).


## How S017 was reached

The earlier 25 MB target is met. S014 could not reach it through frontend work alone. S015 replaced the decoder and generator with a pruned student vocoder that fits and beats S014 on the vocoder-only comparison (1.671 vs 1.771 dB). With the small S015 pitch student it projected to 19,648,867 bytes, but its best full chain was 1.964 dB against S014's 1.862 dB. Retraining that pitch student did not help.

S016 replaced the pitch student with teacher-derived structured-pruned models. Its best candidate reached 2.007 dB / 5.00 Hz, worse than the S015 baseline of 1.964 dB / 4.53 Hz. S016 is closed.

S017 combined the S014 frontend, Kokoro int8 pitch (8,622,252), and the S015 pruned vocoder (9,569,102); with the S018 student (6,076,256 bytes in place of the S014 frontend) that is 24,267,610 compiled-model bytes. The iOS app is only the validation harness; typed text, phoneme entry, UI design, and app-shell size are not goals.

Do not weaken the sound gate, omit required models, download models after install, or shorten speech to meet a target. Do not repeat broad precision sweeps over S014.


## Future work

Target: about **12.1 MB of compiled models** (roughly half of S017) with the same speed and sound gates, including the four-length blind test. Candidates are a smaller pitch stage (S016 showed that naive pruning loses F0 quality), further vocoder compression, and re-testing int4 on the stages that tolerated it. Any attempt must still pass iPhone 12 speed (median RTF 0.30 or lower) and the blind listening rule.

Other open items: test more voices and devices; add an English text-to-phoneme frontend (the models take Kokoro phoneme token ids). The trained student and vocoder checkpoints are published with the models.


## Test method

- Start from the graded Core ML and Swift pipeline of the earlier conversion project [`kokoro-82m-coreml`](https://github.com/qmohitsingh/kokoro-82m-coreml).
- Use the same physical iPhone 12, Release build, sentences, frozen inputs, and `af_heart` voice.
- Measure from token IDs and voice data in to complete 24 kHz waveform out.
- For speed, discard 2 warmups and take the median of 10 runs at each length.
- The app is offline. Remove the benchmark-only ONNX model and ONNX Runtime from the size build.


## Record, but do not grade

- which change produced the size cut: shared weights, fp16, palettization, or another method
- cold start time
- CPU, GPU, and Neural Engine placement
- peak memory
- thermal state during a speed run
- numerical parity, including waveform SNR, DTW log-mel, duration error, and high-frequency energy


## Not part of this project

- adding voices
- integration into another app
- requiring a specific compute unit


## Why these sizes

Kokoro has about 82 million parameters. The weights are about 327 MB in fp32, 164 MB in fp16, 82 MB in int8, and 41 MB in 4-bit, before the app and Swift code. The starting app was about 1.9 GB because each fixed shape stored another copy of the weights and the benchmark also carried ONNX.
