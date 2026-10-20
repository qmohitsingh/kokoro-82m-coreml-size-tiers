"""Download Kokoro Core ML tiers from Hugging Face into artifacts/hf/<tier>/ and verify them.

    python scripts/download_models.py qmohits/kokoro-82m-coreml-size-tiers                       # default tier 25mb (S017)
    python scripts/download_models.py qmohits/kokoro-82m-coreml-size-tiers --tier 50mb
    python scripts/download_models.py qmohits/kokoro-82m-coreml-size-tiers --tier all [--revision REV]
    python scripts/download_models.py qmohits/kokoro-82m-coreml-size-tiers --tier 25mb --verify-only   # no network

Tiers: 500mb (S002), 200mb (S007), 100mb (S011), 50mb (S014 mixed8), 25mb (S017, recommended).
Each tier folder in the repository is self-contained. benchmark/ios/Resources-* link into
artifacts/hf/<tier>/, so xcodegen and xcodebuild work right after this script succeeds.

Every file is checked against MANIFEST.json in this repository (size and SHA-256), so a
changed Hugging Face repository is rejected. The package hash is sha256 over, for each file
sorted by relative path: the relative path, then the file bytes.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def package_sha256(path: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(f.relative_to(path).as_posix().encode())
        with f.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def verify_tier(dest: Path, tier: str, m: dict) -> list[str]:
    errors = []
    base = dest / tier
    for pkg in m["packages"]:
        root = base / pkg["path"]
        if not root.is_dir():
            errors.append(f"{tier}: missing package {pkg['path']}")
            continue
        bad = False
        for entry in pkg["files"]:
            f = root / entry["path"]
            if not f.is_file():
                errors.append(f"{tier}: missing {pkg['path']}/{entry['path']}")
                bad = True
            elif f.stat().st_size != entry["bytes"] or sha256_file(f) != entry["sha256"]:
                errors.append(f"{tier}: size or sha256 mismatch {pkg['path']}/{entry['path']}")
                bad = True
        extra = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} - {
            e["path"] for e in pkg["files"]}
        errors += [f"{tier}: unexpected file {pkg['path']}/{x}" for x in sorted(extra)]
        if not bad and not extra and package_sha256(root) != pkg["package_sha256"]:
            errors.append(f"{tier}: package hash mismatch {pkg['path']}")
    voice = m.get("voice")
    if voice:
        f = base / voice["file"]
        if not f.is_file() or f.stat().st_size != voice["bytes"] or sha256_file(f) != voice["sha256"]:
            errors.append(f"{tier}: voice table mismatch {voice['file']}")
    return errors


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    tiers = list(manifest["tiers_detail"])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_id", help="Hugging Face model repository, qmohits/kokoro-82m-coreml-size-tiers")
    parser.add_argument("--tier", choices=tiers + ["all"], default=manifest["recommended_tier"])
    parser.add_argument("--revision", default=None, help="branch, tag or commit (default: main)")
    parser.add_argument("--dest", type=Path, default=ROOT / "artifacts/hf")
    parser.add_argument("--verify-only", action="store_true", help="skip the download; verify --dest")
    args = parser.parse_args()

    chosen = tiers if args.tier == "all" else [args.tier]
    if not args.verify_only:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=args.repo_id, revision=args.revision, local_dir=args.dest,
                          allow_patterns=[f"{t}/**" for t in chosen])
    failed = False
    for t in chosen:
        m = manifest["tiers_detail"][t]
        errors = verify_tier(args.dest, t, m)
        if errors:
            failed = True
            print(f"VERIFY FAILED ({t})", *errors, sep="\n  ", file=sys.stderr)
        else:
            print(f"OK {t} ({m['experiment']}): {len(m['packages'])} packages match MANIFEST.json "
                  f"(compiled models {m['compiled_models_total_bytes']:,} bytes) in {args.dest / t}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
