# Model card

The model card for the five Core ML tiers (`500mb/`, `200mb/`, `100mb/`, `50mb/`, `25mb/`) and the `af_heart` voice table is `README.md` in the Hugging Face model repository (`qmohits/kokoro-82m-coreml-size-tiers`). It states per-tier sizes (S017, the recommended 25mb tier: 24,267,607 compiled-model bytes, 522,240 voice-table bytes), shapes (fixed for 500mb/200mb/100mb, flexible from 50mb), compute units, speed tables, the objective and listening methodology and their limits, and the upstream Kokoro attribution (`hexgrad/Kokoro-82M`, revision `f3ff3571791e39611d31c381e3a41a3af07b4987`).

Fetch a tier with `scripts/download_models.py qmohits/kokoro-82m-coreml-size-tiers --tier <tier>`, which verifies the files against `MANIFEST.json`. Provenance and licences are in `THIRD_PARTY_NOTICES.md`.
