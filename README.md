# vs_split-models

Model pack catalog for **vs_split**, the stem-separation engine bundled with
[WAVdesk](https://github.com/skxllflower/wavdesk).

WAVdesk never ships model weights in its installer. On first use it fetches `catalog.json` from
this repository, shows the user a confirm dialog with the pack size, and downloads the pack files
from the Release assets listed in the catalog. Every file is verified against the SHA-256 in its
manifest before it is used.

## Layout

- `catalog.json` — the menu: every published pack manifest plus an `updated` stamp. Published as
  an asset on the rolling `catalog` tag so the URL is stable across pack releases.
- `packs/<pack_id>/pack.json` — the manifest source for each pack (copied into the catalog).
- `packs/<pack_id>/LICENSE.txt` — the license text that ships inside the installed pack folder.
- Releases — one per `<pack_id>@<version>`, holding the ONNX files as assets.

## Manifest (`pack.json`)

```json
{
  "id": "demucs-ft",
  "name": "Standard (4 stems)",
  "description": "Vocals, Drums, Bass, Other. Runs on any CPU.",
  "family": "demucs",
  "version": "1.0.0",
  "license": { "spdx": "MIT", "url": "https://github.com/facebookresearch/demucs", "text_file": "LICENSE.txt" },
  "credits": ["Meta AI Research (Demucs v4 / HT-Demucs)", "StemSplit (ONNX export)"],
  "stems": ["vocals", "drums", "bass", "other"],
  "sample_rate": 44100,
  "chunk": { "samples": 343980, "overlap": 0.25, "n_fft": 4096, "hop_samples": 1024, "window": "hann" },
  "files": [
    { "name": "htdemucs_ft_drums.onnx", "sha256": "…", "size": 0, "url": "https://github.com/skxllflower/vs_split-models/releases/download/demucs-ft-1.0.0/htdemucs_ft_drums.onnx" }
  ],
  "capabilities": { "results": ["stems4", "acapella", "instrumental"], "device_pref": "any", "min_vram_mb": 0, "est_rtf": { "cpu": 0.35, "gpu": 0.05 } },
  "min_engine_version": "0.1.0"
}
```

Only packs whose license clearly permits commercial redistribution are published here. The
license text is vendored per pack and the authors are credited in WAVdesk's About page.

## License

The catalog, manifests, and tooling in this repository are MIT. Each model pack carries its own
license in `packs/<pack_id>/LICENSE.txt` and in its Release notes.

## Bring your own RoFormer

`tools/roformer_pack.py` turns a Music-Source-Separation-Training style **Mel-Band RoFormer**
checkpoint (Kim's vocals model and most community vocal models) into a pack folder:

```
python tools/roformer_pack.py model.ckpt config.yaml out/ --id my-vocals --name "My vocals" \
    --license-spdx MIT --license-url https://... --license-file LICENSE.txt --check song.wav
```

It exports the graph without its STFT (WAVdesk's engine computes the transform on the host, pack
kind `roformer_hoststft`, engine 0.3.0 or later), simplifies it, stores the weights as float16,
checks the export against the PyTorch model, and writes `pack.json` + `LICENSE.txt` beside the
`.onnx`. Install that folder from WAVdesk's **Settings → Models → Add Pack from Folder**; it then
appears in every Model list and removes like a downloaded pack. Python needs: torch (CPU is
fine), einops, rotary-embedding-torch, librosa, beartype, pyyaml, onnx, onnxruntime, onnxsim,
onnxconverter-common, soundfile. The vendored model code under `tools/roformer/` is ZFTurbo's
(MIT, `LICENSE.msst.txt`). Only ship packs whose weights you are licensed to redistribute.
