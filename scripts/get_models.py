"""Get the trained networks that the measurement needs, into data_cache/models/.

    python scripts/get_models.py            # the three networks of this project + the cloud mask
    python scripts/get_models.py --no-cloud # only the three networks (the cloud mask needs PyTorch)

1. From this repository's GitHub Release "models-v1", each file checked against its SHA-256 fingerprint
   (a 64-character number computed from the file; any change to the file changes it):
   - water in optical images: U-Net, trained on Sen1Floods11 (CC BY 4.0), all 10 flood events;
   - water in radar images: TerraMind-1.0-base (Apache-2.0), fine-tuned on Sen1Floods11, 3 seeds;
   - burnt land: U-TAE, trained on ImpactMesh-Fire (CC BY 4.0), 3 seeds.
   These networks are released under CC BY 4.0; please credit this repository and the data sets.
2. The cloud mask, OmniCloudMask (Wright et al. 2025), is not re-shared here: its two weight files are
   downloaded from its authors' own Hugging Face page and converted to ONNX on your machine, exactly as
   experiments/b3_export.py does (needs `pip install omnicloudmask==1.8.1 onnx`; OmniCloudMask brings PyTorch).
The construction network is not published (its OSCD training labels are CC BY-NC-SA); without it,
construction questions use the change rule and say so in the answer.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "data_cache" / "models"
RELEASE = "https://github.com/deltasata/satellite-change-agent/releases/download/models-v1/"
FILES = {
    "water_unet_ALL.onnx": "e97a4b727ead283c097f5d134250d65b8c2bd182eb971a19b8e1c4899fcd5d2d",
    "radar_terramind_ALL_s0.onnx": "7a8b7f3183ffb8e644401e77d3e7b6cf21e9c3e98aed860f7cf35cd365327ce4",
    "radar_terramind_ALL_s0.onnx.data": "31cdaeed4a2d8bc5ab9af59b67d67b467810d64cb6be83c70ae7a30673f06b57",
    "radar_terramind_ALL_s1.onnx": "eed2fd6be7b84f1cb95f4d07b159fc1fa58f6346aeb04433b4d59810233deeff",
    "radar_terramind_ALL_s1.onnx.data": "8650f596745e4edf3383cd0da7c4a4ab22edb75f3804862387ba8724698ab067",
    "radar_terramind_ALL_s2.onnx": "37f373844bf91357ac01508701bbf72a0017757831f0e892c6b846b1da82a4c1",
    "radar_terramind_ALL_s2.onnx.data": "db81552c839920506c82c78309de9685d59f3761bb700957d7adbc604255c8dc",
    "burn_utae_s0.onnx": "53491bbdb021878a658a0f6ccd599a226b8b911241cdf99716812c8abaebab44",
    "burn_utae_s1.onnx": "bf8a5a1ecfcfab180a3b9abdc3ee6d8a4695d47ba346fa90b1d38ca8bd8566b9",
    "burn_utae_s2.onnx": "e40ff2a025f7397ea4a1084a8ff512cd160d550db65b3734ef733bfdf91fcfdb",
}
OCM = {"regnety_004": ("tu-regnety_004", "PM_model_OCM_7.97_R_G_NIR_3_smp_regnety_004.pycls_in1k_PT_state"),
       "edgenext_small": ("tu-edgenext_small", "PM_model_OCM_7.97_R_G_NIR_3_smp_edgenext_small.usi_in1k_PT_state")}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def get_networks() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    for name, want in FILES.items():
        dest = MODELS / name
        if dest.exists() and sha256(dest) == want:
            print(f"  {name}: present")
            continue
        tmp = dest.with_suffix(dest.suffix + ".part")
        print(f"  {name}: downloading ...", flush=True)
        urllib.request.urlretrieve(RELEASE + name, tmp)
        got = sha256(tmp)
        if got != want:
            tmp.unlink()
            sys.exit(f"{name}: fingerprint {got} is not the expected {want}; nothing was kept")
        tmp.rename(dest)


def get_cloud_mask() -> None:
    import torch
    from huggingface_hub import hf_hub_download
    from omnicloudmask.model_utils import load_model_from_weights
    for short, (timm_name, stem) in OCM.items():
        dest = MODELS / f"ocm_v4_{short}.onnx"
        if dest.exists():
            print(f"  {dest.name}: present")
            continue
        print(f"  {dest.name}: downloading the authors' weights and converting ...", flush=True)
        weights = hf_hub_download(repo_id="NickWright/OmniCloudMask", filename=f"{stem}.safetensors",
                                  local_dir=MODELS / "omnicloudmask")
        net = load_model_from_weights(model_name=timm_name, weights_path=weights, model_library="smp",
                                      device=torch.device("cpu"), dtype=torch.float32)
        torch.onnx.export(net, (torch.randn(1, 3, 512, 512),), str(dest), input_names=["x"], output_names=["logits"],
                          dynamic_axes={"x": {0: "n", 2: "h", 3: "w"}, "logits": {0: "n", 2: "h", 3: "w"}},
                          opset_version=17, dynamo=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cloud", action="store_true", help="skip the cloud mask (it needs PyTorch)")
    a = ap.parse_args()
    print("Networks of this project (GitHub Release models-v1):")
    get_networks()
    if not a.no_cloud:
        print("Cloud mask (OmniCloudMask, from its authors' page):")
        get_cloud_mask()
    print("done:", MODELS)
