# Retrain the 25 MB student on your own text

The recipe behind the S018 student. You give it plain English text and a list of the sounds or words the student gets wrong. It builds a training corpus weighted toward them, caches the teacher's outputs, fine-tunes the student, converts it to Core ML and scores it. The scripts are `scripts/s018_*.py`; run everything from the repository root. They reproduce the S018 method with its default hyperparameters.

## What you need

Python packages: `torch`, `coremltools`, `onnxruntime`, `numpy`, `kokoro` 0.9.4 and `misaki[en]` (the text-to-phoneme frontend, with espeak). Also `ffmpeg`, and a whisper.cpp build (`whisper-cli`). Core ML conversion needs macOS with Xcode command line tools (`xcrun coremlcompiler`). Training runs on `mps` (Apple silicon) or `cpu`; pass `--device cpu` where needed.

Public sources only:

| File | Where it comes from |
|---|---|
| Teacher weights `kokoro-v1_0.pth` and `config.json`, to `models/pytorch/` | Hugging Face `hexgrad/Kokoro-82M` (see "Reproduce the voice table" in the main README) |
| Init checkpoint `student-w256-b-step10000.pt` | this project's Hugging Face model repository `qmohits/kokoro-82m-coreml-size-tiers`, folder `checkpoints/` |
| Pitch and vocoder packages, voice table `af_heart_f32.bin` | the same repository, folder `25mb/` (`scripts/download_models.py`) |
| Teacher ONNX (scoring reference only) | `kokoro-v1.0.onnx` from the kokoro-onnx project's v1.0 model files (Hugging Face `onnx-community/Kokoro-82M-v1.0-ONNX` also works; the script reads the input name from the file) |
| Whisper model | `ggml-medium.en.bin` from the whisper.cpp model repository `ggerganov/whisper.cpp` on Hugging Face |

Put the voice table where the scripts expect it: `<voice-dir>/<voice>_f32.bin` (510 x 256 float32). The default voice is `af_heart`; every script takes `--voices` to add more, but a voice needs its own cached teacher rows and the vocoder was trained for `af_heart` only.

Text: any plain-text files, public domain or your own. You need two disjoint sets: text for training and text for the held-out set. Use different books or documents; never let held-out sentences into training (the corpus builder also removes exact and substring matches).

## Steps

Below, `W=runs-s018` is the working directory. Every script takes `--work`; nothing is written elsewhere.

1. **Held-out set first, then freeze it.** About 70 sentences that cover every phoneme English text can produce at least 5 times.

   ```
   python scripts/s018_heldout.py --work $W --text heldout_book1.txt heldout_book2.txt
   ```

   Writes `$W/heldout/{sentences.txt, tokens.json, counts.json, SHA256}`. Add `--exclude my_test_sentences.txt` for any other test set. Small inputs cannot reach 5 occurrences per token: lower `--min-count` and `--min-sentences`.

2. **Corpus.** Copy `scripts/s018_corpus_config.example.json` and edit it: `target_tokens` (phoneme symbol and the count you want), `problem_words` (words the student mispronounces, optional), `names`, `focus_groups` (word lists for one sound, optional), the sizes of the targeted and general portions, `seed`, and `exclude` (extra files of test sentences).

   ```
   python scripts/s018_corpus.py --work $W --text train_book1.txt train_book2.txt --config my_corpus_config.json
   ```

   Writes `$W/corpus/{train.jsonl, stats.json, dedupe.json}`. Check `stats.json` (`target_token_counts` against `target_token_wanted`, `problem_word_counts`) and `dedupe.json` (all `final_*_matches` must be 0). Targeted sentences come from neutral built-in everyday templates, your problem words, the focus groups, and the sentences of your text densest in the target tokens; the general portion is sampled from your text. The more varied text you give it, the better the general portion.

3. **Cache the teacher** (once per corpus, per voice).

   ```
   python scripts/s018_cache_teacher.py train   --work $W --voice-dir voices
   python scripts/s018_cache_teacher.py heldout --work $W --voice-dir voices
   ```

4. **Train.** Start from the published checkpoint. The default is stage A (feature losses only, about 0.1 s per step on `mps`); `--stage b` adds pitch and log-mel terms through the frozen teacher decoder and costs about 10 times more per step.

   ```
   python scripts/s018_train.py --work $W --voice-dir voices --init student-w256-b-step10000.pt \
       --name run1 --steps 3000 --targets ʧ ʤ A I O W Y --w-target 4 --lr 2e-4
   ```

   `--targets` takes phoneme symbols, optionally `SYM=weight` for a per-token weight (default `--w-target 4`). Checkpoints go to `$W/runs/run1/step<N>.pt`; `log.jsonl` has the validation numbers on the held-out cache for all tokens and for the target tokens. The S018 student used `--stage b --down-count 4 --lr 2e-4 --steps 3000` on a 13,600-sentence corpus with a 5,200 / 1,400 / 7,000 targeted / focus / general split.

5. **Convert to Core ML.**

   ```
   python scripts/s018_convert.py --checkpoint $W/runs/run1/step3000.pt --out $W/runs/run1
   ```

   Writes `kokoro_student_w256.mlpackage` and `size.json`. The package must have the same input and output names, shapes and dtypes as the published `kokoro_student_w256.mlpackage` (`input_ids` int32 `[1, 1..512]`, `style` fp32 `[1, 256]`; outputs `duration_hidden`, `duration_logits`, `text_features`), so the pitch and vocoder packages and the Swift chain work unchanged. `s018_score.py` checks this when you pass `--baseline`.

6. **Score: WER against the teacher on the held-out set.** This is the gate.

   ```
   python scripts/s018_score.py --work $W --name run1 --voice-dir voices \
       --teacher-onnx kokoro-v1.0.onnx --pitch 25mb/kokoro_pitch_int8_flex.mlpackage \
       --vocoder 25mb/kokoro_s015_vocoder_int8.mlpackage \
       --whisper-cli whisper-cli --whisper-model ggml-medium.en.bin \
       --baseline 25mb/kokoro_student_w256.mlpackage
   ```

   It renders the teacher (`ref`), the student you started from (`base`) and the new one (`cand`) through the same chain, transcribes with whisper and prints the word error rate of each against the text. Pass when `cand` is within 1 point of `ref` on the held-out set, and no word the teacher says correctly fails in more than one sentence (`repeat_word_violations`). Look at `fixed`, `newly_broken` and `ref_ok_cand_wrong` in `score.json` for the words.

7. **Optional: DTW log-mel** on 5 frozen benchmark items and 10 held-out sentences. Record it; do not gate on it (see below).

   ```
   python scripts/s018_dtw.py --work $W --name run1 --package $W/runs/run1/kokoro_student_w256.mlpackage \
       --student-ckpt student-w256-b-step10000.pt --pitch 25mb/kokoro_pitch_int8_flex.mlpackage \
       --vocoder 25mb/kokoro_s015_vocoder_int8.mlpackage
   ```

## What can go wrong

- **You will not hear it, and log-mel DTW will not show it.** The S017 student mispronounced common words (the "ch" and "j" sounds and long vowels such as the "ay" in "say" and the "i" in "nice") and still passed blind listening and the 1.86 dB DTW bar. Gate on Whisper WER against the teacher on a held-out set the training never saw. Whisper hears the wrong word where a listener hears fluent speech.
- **DTW can get worse while the words get better.** In S018 the student's DTW went from 1.79 to 2.58 dB (from 3.12 dB with feature losses only) while WER on the held-out set fell from about 20% to under 4%. Read DTW as a regression report on the audio, not a pass/fail test. The `--stage b` log-mel terms halve the regression; they do not remove it.
- **Weight the weak tokens in the loss.** Sounds are rare in running text. Average the loss over frames and the weak ones disappear in the mean. The recipe weights each target token 4x (`--w-target`) and normalizes by the sum of token weights, so every token counts once. The same goes for the corpus: add targeted sentences until each weak token has thousands of occurrences (`target_token_counts` in `stats.json`).
- **Check per-token losses, not only the average.** The overall text and hidden SNR in `log.jsonl` can improve while one phoneme stays bad. Compare the `target` row with the `all` row, and when a word is still wrong, look at which tokens it uses and add them to `--targets`. S018 found the diphthong in "boy" and "voice" missing from the first target list this way: it was added in the second run.
- **Hold out text the corpus never saw.** If the problem words are in the held-out sentences and the corpus, WER only measures memory. The corpus builder drops matching sentences, but it cannot know about your other test sets: list them under `exclude`.
- **A student trained on one voice is tested on one voice.** The vocoder was trained for `af_heart`. A second voice needs its own cached rows and its own WER gate.
- **If your app stores the audio, do not store it at a low bitrate.** AAC at 24 kbps cuts everything above about 7.5 kHz, and the speech sounds fuzzy even though the student and vocoder are fine (the band above 9 kHz drops by about 65 dB). Use 48 kbps or more (64 kbps keeps the full band), or store lossless.
- **Training is not byte-reproducible.** Seeds are fixed, but `mps` kernels differ between machines and torch versions, so expect the same trend, not the same numbers.
