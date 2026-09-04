"""Turn a Mel-Band RoFormer checkpoint into a vs_split pack folder.

    python tools/roformer_pack.py <checkpoint.ckpt> <config.yaml> <out_dir> \\
        --id my-vocals --version 1.0.0 --name "My vocals model" \\
        [--description "..."] [--license-spdx MIT --license-url URL --license-file LICENSE.txt] \\
        [--credits "..."] [--release-base https://github.com/<you>/<repo>/releases/download/<tag>/] \\
        [--fp32] [--check some_44100_stereo.wav]

The checkpoint is a Music-Source-Separation-Training (ZFTurbo) style Mel-Band
RoFormer (`models/bs_roformer/mel_band_roformer.py`) with its YAML config;
Kim's vocals model and most community vocal models are this shape. The graph
is exported WITHOUT its STFT (the engine computes the transform on the host:
pack impl kind `roformer_hoststft`), simplified, and its weights stored as
float16, then the folder gets pack.json + LICENSE.txt. That folder is exactly
what WAVdesk's Settings → Models → "Add Pack from Folder" installs, and what
this repo publishes as a release for the catalog.

Needs: torch (CPU is fine), einops, rotary-embedding-torch, librosa, beartype,
pyyaml, onnx, onnxruntime, onnxsim, onnxconverter-common, numpy, soundfile.
"""
import argparse, hashlib, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "roformer"))
import numpy as np
import torch
import yaml
from einops import rearrange

KIND = "roformer_hoststft"

def build_model(ckpt_path, cfg_path):
    from mel_band_roformer import MelBandRoformer
    cfg = yaml.unsafe_load(open(cfg_path))
    m = dict(cfg["model"])
    for k in ("multi_stft_resolution_loss_weight", "multi_stft_resolutions_window_sizes",
              "multi_stft_hop_size", "multi_stft_normalized"):
        m.pop(k, None)
    m["flash_attn"] = False
    model = MelBandRoformer(**m)
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise SystemExit(f"checkpoint does not match the config: missing {missing[:5]}")
    if unexpected:
        print(f"note: {len(unexpected)} unexpected keys ignored (e.g. {unexpected[:3]})")
    model.eval()
    return model, cfg

class Body(torch.nn.Module):
    """model.forward between the STFT and the iSTFT, in real arithmetic: the
    band gather, the transformer stack, the mask heads, the band scatter as
    a MatMul with a one-hot matrix, the complex multiply, the DC zeroing."""
    def __init__(self, model):
        super().__init__()
        self.m = model
        FS = int(model.freq_indices.max().item()) + 1
        K = int(model.freq_indices.numel())
        M = torch.zeros(K, FS)
        M[torch.arange(K), model.freq_indices] = 1.0
        self.register_buffer("M", M)
        denom = model.num_bands_per_freq.repeat_interleave(model.audio_channels).float()
        self.register_buffer("inv_denom", (1.0 / denom.clamp(min=1e-8)).view(1, 1, FS, 1, 1))
        dc = torch.ones(FS)
        if model.zero_dc:
            dc[: model.audio_channels] = 0.0
        self.register_buffer("dc_mask", dc.view(1, 1, FS, 1, 1))
        self.FS, self.K = FS, K

    def forward(self, spec):
        m = self.m
        x = spec[:, m.freq_indices]
        x = rearrange(x, "b f t c -> b t (f c)")
        x = m.band_split(x)
        for block in m.layers:
            time_tr, freq_tr = block[-2], block[-1]
            x = rearrange(x, "b t f d -> b f t d")
            b, f, t, d = x.shape
            x = time_tr(x.reshape(b * f, t, d)).reshape(b, f, t, d)
            x = rearrange(x, "b f t d -> b t f d")
            x = freq_tr(x.reshape(b * t, f, d)).reshape(b, t, f, d)
        masks = torch.stack([fn(x) for fn in m.mask_estimators], dim=1)
        masks = rearrange(masks, "b n t (f c) -> b n t c f", c=2)
        b, n, t, c, k = masks.shape
        summed = torch.matmul(masks.reshape(b * n * t * c, k), self.M).reshape(b, n, t, c, self.FS)
        summed = summed.permute(0, 1, 4, 2, 3)
        ma = summed * self.inv_denom
        sr, si = spec[..., 0].unsqueeze(1), spec[..., 1].unsqueeze(1)
        mr, mi = ma[..., 0], ma[..., 1]
        out = torch.stack((sr * mr - si * mi, sr * mi + si * mr), dim=-1)
        return out * self.dc_mask

def host_stft(model, audio):
    w = model.stft_window_fn()
    z = torch.stft(audio, **model.stft_kwargs, window=w, return_complex=True)
    return rearrange(torch.view_as_real(z), "s f t c -> 1 (f s) t c").contiguous()

def host_istft(model, spec_out, length):
    w = model.stft_window_fn()
    S = model.audio_channels
    z = rearrange(spec_out, "b n (f s) t c -> (b n s) f t c", s=S)
    z = torch.view_as_complex(z.contiguous())
    y = torch.istft(z, **model.stft_kwargs, window=w, return_complex=False, length=length)
    return rearrange(y, "(n s) t -> n s t", s=S)

def sha256_of(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt"); ap.add_argument("config"); ap.add_argument("out_dir")
    ap.add_argument("--id", required=True); ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--name", required=True); ap.add_argument("--description", default="")
    ap.add_argument("--license-spdx", default="unknown"); ap.add_argument("--license-url", default="")
    ap.add_argument("--license-file", help="the model's license text, copied into LICENSE.txt")
    ap.add_argument("--credits", action="append", default=[])
    ap.add_argument("--release-base", default="", help="URL prefix for files[].url (catalog packs)")
    ap.add_argument("--fp32", action="store_true", help="keep float32 weights (twice the size)")
    ap.add_argument("--keep-fp32", action="store_true", help="also keep the float32 graph beside the fp16 one")
    ap.add_argument("--check", help="a 44.1 kHz stereo WAV: parity check of the export against the model")
    ap.add_argument("--est-rtf-cpu", type=float, default=1.5)
    ap.add_argument("--est-rtf-gpu", type=float, default=0.15)
    a = ap.parse_args()

    model, cfg = build_model(a.ckpt, a.config)
    chunk = int(cfg["audio"]["chunk_size"])
    sr = int(cfg["audio"]["sample_rate"])
    if sr != 44100:
        raise SystemExit(f"the engine runs packs at 44.1 kHz; this config says {sr}")
    hop = int(model.stft_kwargs["hop_length"]); n_fft = int(model.stft_kwargs["n_fft"])
    if model.stft_kwargs.get("win_length", n_fft) != n_fft:
        raise SystemExit("win_length != n_fft is not supported")
    body = Body(model).eval()
    T = 1 + chunk // hop
    print(f"chunk {chunk} samples, {body.FS} rows x {T} frames, {body.K} band rows", flush=True)

    # Stems: the config's instruments, or its single target.
    tr = cfg.get("training", {})
    target = tr.get("target_instrument")
    stems = [target] if target else list(tr.get("instruments", []))
    if not stems:
        raise SystemExit("config names no target_instrument / instruments")
    if len(stems) != len(model.mask_estimators):
        raise SystemExit(f"config names {len(stems)} stems but the model has {len(model.mask_estimators)} heads")

    os.makedirs(a.out_dir, exist_ok=True)
    stem_tag = "-".join(stems) if len(stems) <= 2 else f"{len(stems)}stems"
    base = f"{a.id}_{stem_tag}_hoststft"
    fp32_path = os.path.join(a.out_dir, base + "_fp32.onnx")
    final_path = os.path.join(a.out_dir, base + ("_fp32.onnx" if a.fp32 else "_fp16weights.onnx"))

    if a.check:
        import soundfile as sf
        x, fsr = sf.read(a.check, dtype="float32", always_2d=True)
        if fsr != sr:
            raise SystemExit(f"--check must be {sr} Hz")
        x = x[:chunk].T.copy()
        if x.shape[1] < chunk:
            x = np.pad(x, ((0, 0), (0, chunk - x.shape[1])))
        audio = torch.from_numpy(x)
    else:
        audio = torch.randn(2, chunk) * 0.05
    with torch.no_grad():
        ref = model(audio.unsqueeze(0))
        spec = host_stft(model, audio)
        out = body(spec)
        recon = host_istft(model, out, chunk)
    ref = ref if ref.ndim == 4 else ref.unsqueeze(1)          # [1, N, S, L]
    d = float((recon - ref[0]).abs().max()); rms = float(ref.pow(2).mean().sqrt())
    print(f"host pipeline vs model: max abs {d:.2e} (rms {rms:.4f})", flush=True)
    if d > 1e-3 * max(1.0, rms) + 1e-5:
        raise SystemExit("the host STFT contract does not reproduce this model")

    t0 = time.time()
    torch.onnx.export(body, (spec,), fp32_path, input_names=["spec"], output_names=["spec_out"],
                      opset_version=17, dynamo=False, do_constant_folding=True)
    print(f"exported fp32 {os.path.getsize(fp32_path) >> 20} MB in {time.time() - t0:.0f}s", flush=True)

    import onnx
    graph_path = fp32_path
    try:
        from onnxsim import simplify
        m = onnx.load(fp32_path)
        ms, ok = simplify(m)
        if ok:
            onnx.save(ms, fp32_path)
            print(f"simplified: {len(m.graph.node)} -> {len(ms.graph.node)} nodes", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"note: onnxsim skipped ({e})", flush=True)
    if not a.fp32:
        try:
            from onnxconverter_common import float16
            m = onnx.load(fp32_path)
            # Reductions, divisions and the exp/log family stay float32: the
            # RMSNorm sums of squares over the RAW spectrogram rows overflow
            # float16 on DirectML (NaN out), while the matmuls, which are the
            # cost, run in float16. ORT's CPU provider upcasts anyway.
            m16 = float16.convert_float_to_float16(
                m, keep_io_types=True,
                op_block_list=list(float16.DEFAULT_OP_BLOCK_LIST) + [
                    "ReduceL2", "ReduceSum", "ReduceMean", "ReduceMax", "Pow", "Sqrt", "Div",
                    "Clip", "Softmax", "Exp", "Log", "Erf", "Sigmoid",
                ])
            onnx.save(m16, final_path)
            graph_path = final_path
            print(f"fp16 weights: {os.path.getsize(final_path) >> 20} MB", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"note: fp16 conversion failed ({e}); keeping fp32", flush=True)
            final_path = fp32_path
    if graph_path != fp32_path and os.path.exists(fp32_path) and not a.keep_fp32:
        os.remove(fp32_path)

    import onnxruntime as ort
    sess = ort.InferenceSession(final_path, providers=["CPUExecutionProvider"])
    t0 = time.time()
    got = sess.run(["spec_out"], {"spec": spec.numpy()})[0]
    dt = time.time() - t0
    recon2 = host_istft(model, torch.from_numpy(got), chunk)
    d2 = float((recon2 - ref[0]).abs().max())
    print(f"onnx end-to-end vs model: max abs {d2:.2e}; cpu {dt:.1f}s per {chunk / sr:.1f}s chunk", flush=True)
    if d2 > 5e-3 * max(1.0, rms):
        raise SystemExit("the exported graph does not reproduce the model")

    # LICENSE.txt: a notice plus the model's own license text.
    lic_path = os.path.join(a.out_dir, "LICENSE.txt")
    notice = (f"NOTICE\nThis pack contains an ONNX export of the model '{a.name}' ({a.id} {a.version}).\n"
              f"The export keeps the spectral transform on the host (vs_split pack kind {KIND}).\n"
              + ("Weights are stored as float16 and computed in float32.\n" if not a.fp32 else "")
              + f"License: {a.license_spdx}" + (f" ({a.license_url})" if a.license_url else "") + "\n\n")
    text = open(a.license_file, encoding="utf-8").read() if a.license_file else ""
    with open(lic_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(notice + text)

    files = []
    for name in (os.path.basename(final_path), "LICENSE.txt"):
        p = os.path.join(a.out_dir, name)
        entry = {"name": name, "sha256": sha256_of(p), "size": os.path.getsize(p),
                 "url": (a.release_base + name) if a.release_base else ""}
        if name.endswith(".onnx"):
            entry["stem"] = stems[0] if len(stems) == 1 else ""
            if len(stems) > 1:
                entry["stems"] = stems
        files.append(entry)
    virtual = ["instrumental"] if stems == ["vocals"] else []
    manifest = {
        "schema": 1, "id": a.id, "name": a.name,
        "description": a.description or f"{', '.join(s.capitalize() for s in stems)}. Mel-Band RoFormer.",
        "family": "roformer", "version": a.version,
        "license": {"spdx": a.license_spdx, "url": a.license_url, "text_file": "LICENSE.txt"},
        "credits": a.credits, "stems": stems, "virtual_stems": virtual, "sample_rate": sr,
        "impl": {"kind": KIND, "input": "spec", "output": "spec_out",
                 "inputs": ["spec"], "outputs": ["spec_out"],
                 "segment_samples": chunk, "overlap": 0.5, "stems_order": stems,
                 "stft": {"n_fft": n_fft, "hop": hop, "normalized": bool(model.stft_kwargs.get("normalized", False))}},
        "files": files,
        "results": (["acapella", "instrumental"] if stems == ["vocals"] else ["stems4"]),
        "device_pref": "any", "min_vram_mb": 0,
        "est_rtf": {"cpu": a.est_rtf_cpu, "gpu": a.est_rtf_gpu},
        "min_engine_version": "0.3.0",
        "total_bytes": sum(f["size"] for f in files),
    }
    with open(os.path.join(a.out_dir, "pack.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2); f.write("\n")
    print(f"pack folder ready: {a.out_dir} ({manifest['total_bytes'] >> 20} MB)", flush=True)

if __name__ == "__main__":
    main()
