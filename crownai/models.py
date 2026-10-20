"""Neural network files crownai uses: where they live, download and checksum.

Files come from this repository's GitHub releases (converted to ONNX, with the
original licence texts next to them).  A download is checked against its
SHA-256 before it is used.  The folder defaults to ``~/.crownai/models`` and can
be set with the ``CROWNAI_MODELS`` environment variable.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

RELEASE = "https://github.com/Tony21213/Tony21213/releases/download/photo-v1/"
MODELS = {
    # name: (file in the release, path under the models folder, size, sha256, licence)
    "dentalmodelseg": ("dentalmodelseg-v3.0.onnx", "dentalmodelseg/model.onnx", 6_632_040,
                       "8c3aaf7c381c9c84a0f24aa9dad007c5da29f8869293d020d89a21bbd847e82c",
                       "Apache-2.0 (DCBIA-OrthoLab SlicerDentalModelSeg)"),
}


def models_dir() -> Path:
    return Path(os.environ.get("CROWNAI_MODELS") or Path.home() / ".crownai" / "models")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure(name: str, folder: str | Path | None = None, log=print) -> Path:
    """Path to the model file, downloading and verifying it on first use."""
    release_file, rel_path, size, sha, _licence = MODELS[name]
    folder = Path(folder) if folder else models_dir()
    path = folder / rel_path
    if path.exists() and path.stat().st_size == size and _sha256(path) == sha:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    log(f"downloading {name} ({size / 1e6:.1f} MB)")
    urllib.request.urlretrieve(RELEASE + release_file, tmp)
    if _sha256(tmp) != sha:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{name}: checksum mismatch, download discarded")
    tmp.replace(path)
    return path


def ensure_dentalmodelseg(folder: str | Path | None = None, log=print) -> Path:
    """The models folder holding ``dentalmodelseg/model.onnx`` (as crownai.scan_seg expects)."""
    return ensure("dentalmodelseg", folder, log).parent.parent
