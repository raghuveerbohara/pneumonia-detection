"""Restore the exact trained model from browser-upload-sized pieces."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore_model():
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
    target = MODEL_DIR / "best_model.keras"
    if target.exists() and sha256_file(target) == manifest["model_sha256"]:
        return target, manifest

    partial = MODEL_DIR / "best_model.partial"
    try:
        with partial.open("wb") as output:
            for part in manifest["parts"]:
                source = MODEL_DIR / part["filename"]
                if source.stat().st_size != part["bytes"] or sha256_file(source) != part["sha256"]:
                    raise ValueError(f"Model piece is incomplete or changed: {source.name}")
                with source.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        output.write(chunk)
        if sha256_file(partial) != manifest["model_sha256"]:
            raise ValueError("Restored model checksum does not match the trained model.")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target, manifest


if __name__ == "__main__":
    path, metadata = restore_model()
    print(f"Model restored and verified: {path.name}")
    print(f"SHA-256: {metadata['model_sha256']}")
