"""Build packs/<id>/pack.json + LICENSE.txt and regenerate catalog.json.

usage: python tools/make_pack.py demucs-ft <dir-with-model-files> <version>
The model files must already be in <dir>; hashes/sizes are computed here. Asset URLs point at the
GitHub Release tag "<id>-<version>" of this repo. Run from the repo root.
"""
import hashlib
import json
import os
import sys
from datetime import date
from pathlib import Path

REPO = "skxllflower/vs_split-models"
ROOT = Path(__file__).resolve().parent.parent

PACKS = {
    "demucs-ft": {
        "name": "Standard (4 stems)",
        "description": "Vocals, Drums, Bass, Other. Meta's fine-tuned HT-Demucs. Runs on any CPU.",
        "family": "demucs",
        "license": {"spdx": "MIT", "url": "https://github.com/facebookresearch/demucs", "text_file": "LICENSE.txt"},
        "credits": [
            "Meta AI Research: HT-Demucs (Rouard, Massa, Defossez, ICASSP 2023)",
            "StemSplit: ONNX export (huggingface.co/StemSplitio/htdemucs-ft-onnx)",
        ],
        "stems": ["vocals", "drums", "bass", "other"],
        "virtual_stems": ["instrumental"],
        "sample_rate": 44100,
        "impl": {"kind": "demucs_waveform", "input": "mix", "output": "stems",
                 "segment_samples": 343980, "overlap": 0.25,
                 "stems_order": ["drums", "bass", "other", "vocals"]},
        "model_files": [
            ("htdemucs_ft_vocals_fp16weights.onnx", "vocals"),
            ("htdemucs_ft_drums_fp16weights.onnx", "drums"),
            ("htdemucs_ft_bass_fp16weights.onnx", "bass"),
            ("htdemucs_ft_other_fp16weights.onnx", "other"),
        ],
        "results": ["acapella", "instrumental", "stems4"],
        "device_pref": "any", "min_vram_mb": 0,
        "est_rtf": {"cpu": 0.6, "gpu": 0.1},
        "min_engine_version": "0.1.0",
        "license_notice": (
            "NOTICE\n"
            "This pack contains ONNX exports of Meta's HT-Demucs fine-tuned models (htdemucs_ft),\n"
            "released by Meta Platforms, Inc. under the MIT License reproduced below.\n"
            "The ONNX conversion was published by StemSplit (huggingface.co/StemSplitio/htdemucs-ft-onnx)\n"
            "under the MIT License. Weights are stored as float16 and computed in float32.\n"
            "Packaged for WAVdesk's vs_split engine by Vacant Systems.\n\n"
        ),
    },
}


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    pack_id, src_dir, version = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    spec = PACKS[pack_id]
    out_dir = ROOT / "packs" / pack_id
    out_dir.mkdir(parents=True, exist_ok=True)

    upstream_license = (src_dir / "LICENSE.demucs.txt").read_text(encoding="utf-8")
    lic_path = out_dir / "LICENSE.txt"
    lic_path.write_text(spec["license_notice"] + upstream_license, encoding="utf-8", newline="\n")

    base = f"https://github.com/{REPO}/releases/download/{pack_id}-{version}/"
    files = []
    for name, stem in spec["model_files"]:
        p = src_dir / name
        files.append({"name": name, "stem": stem, "sha256": sha256_of(p), "size": p.stat().st_size, "url": base + name})
    files.append({"name": "LICENSE.txt", "sha256": sha256_of(lic_path), "size": lic_path.stat().st_size, "url": base + "LICENSE.txt"})

    manifest = {
        "schema": 1, "id": pack_id, "name": spec["name"], "description": spec["description"],
        "family": spec["family"], "version": version, "license": spec["license"], "credits": spec["credits"],
        "stems": spec["stems"], "virtual_stems": spec["virtual_stems"], "sample_rate": spec["sample_rate"],
        "impl": spec["impl"], "files": files, "results": spec["results"], "device_pref": spec["device_pref"],
        "min_vram_mb": spec["min_vram_mb"], "est_rtf": spec["est_rtf"], "min_engine_version": spec["min_engine_version"],
        "total_bytes": sum(f["size"] for f in files),
    }
    (out_dir / "pack.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")

    # Regenerate the catalog from every packs/*/pack.json (newest version per id wins by string compare).
    packs = []
    for pj in sorted(ROOT.glob("packs/*/pack.json")):
        packs.append(json.loads(pj.read_text(encoding="utf-8")))
    catalog = {
        "schema": 1, "updated": date.today().isoformat(),
        "catalog_url": f"https://github.com/{REPO}/releases/download/catalog/catalog.json",
        "packs": packs,
    }
    (ROOT / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out_dir / 'pack.json'} ({manifest['total_bytes']} bytes across {len(files)} files) and catalog.json with {len(packs)} pack(s)")


if __name__ == "__main__":
    main()
