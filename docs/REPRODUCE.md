# Reproducing the results

Every table in [METHODS.md](METHODS.md) can be recomputed with the commands below. Install the training environment first (one 24 GB GPU per model; several runs in parallel on several GPUs):

```bash
python -m venv .venv && source .venv/bin/activate && pip install -e .     # pipeline (CPU)
python -m venv .venv-ml && .venv-ml/bin/pip install -e ".[ml]"            # training (one GPU)
python -m venv .venv-fire && .venv-fire/bin/pip install impactmesh==0.2.0 terratorch==1.2.14   # TerraMind-base-Fire only
```

The networks used by the pipeline can be exported to ONNX after training (`experiments/b1_export.py`, `b2_export`, `b3_export`, `b4_export --model terramind`, `b8_export`), or downloaded with `python scripts/get_models.py` (all except the construction network).

Reproducing the results (water: about 4 hours on four 24 GB GPUs; clouds: about 3 hours more;
construction: about 2.5 hours more; burnt land: about 2.5 hours more; every clear date: about 4 hours more):

```bash
python -m core.data.sen1floods11 && python -m core.data.sen1floods11_l2a && python -m core.data.sen1floods11_s1rtc
python -m experiments.b1_rules && .venv-ml/bin/python -m experiments.b1_train --run-all --gpus 0,1,2,3
python -m experiments.b2_rules && .venv-ml/bin/python -m experiments.b2_train --plan A,B-select --gpus 0,1,2,3
.venv-ml/bin/python -m experiments.b2_report A        # radar table
.venv-ml/bin/python -m experiments.b2_train --plan B-final && .venv-ml/bin/python -m experiments.b2_eval
.venv-ml/bin/python -m experiments.b2_report B-final  # comparison with published numbers
python -m experiments.b1_report                       # optical table
# the networks the agent uses for new places (trained on every flood event), then export to ONNX:
.venv-ml/bin/python -m experiments.b1_train --model unet --event ALL
.venv-ml/bin/python -m experiments.b2_train --plan A-all --gpus 0,1,2
# clouds (CloudSEN12+: download the 9 needed files, 153 GB, into data_cache/cloudsen12/taco/ first)
.venv-ml/bin/python -m core.data.cloudsen12 && python -m core.data.cloudsen12_es
python -m experiments.b3_rules && .venv-ml/bin/python -m experiments.b3_ocm
.venv-ml/bin/python -m experiments.b3_train --all --gpus 0,1,2 && python -m experiments.b3_report A
.venv-ml/bin/python -m experiments.b3_export && .venv-ml/bin/python -m experiments.b3_check
python -m experiments.b3_exam run scl && python -m experiments.b3_exam run ocm && python -m experiments.b3_exam report
# construction (OSCD's original files from the authors' site; Planetary Computer images, about 3 GB)
python -m core.data.oscd_pc download && python -m core.data.oscd_pc check && python -m core.data.oscd_pc
python -m experiments.b4_common && python -m experiments.b4_rules
.venv-ml/bin/python -m experiments.b4_train --all --gpus 0,1,2,3 && python -m experiments.b4_report
.venv-ml/bin/python -m experiments.b4_export --model terramind
for p in B C; do python -m experiments.b4_exam run rule $p && python -m experiments.b4_exam run terramind $p; done
python -m experiments.b4_exam report terramind
# burnt land (ImpactMesh-Fire labels, 50 MB; Earth Search images read on four GPUs, about 1.5 hours)
.venv-ml/bin/python -m core.data.impactmesh_es download && .venv-ml/bin/python -m core.data.impactmesh_es
.venv-ml/bin/python -m core.data.impactmesh_es check     # alignment with the data set's own images
.venv-ml/bin/python -m experiments.b5_common && .venv-ml/bin/python -m experiments.b5_common pack
.venv-ml/bin/python -m experiments.b5_rules && .venv-ml/bin/python -m experiments.b5_train --all --gpus 0,1,2
# Prithvi-EO-2.0-300M-BurnScars: put the model files from Hugging Face in data_cache/models/prithvi_burnscars
.venv-ml/bin/python -m experiments.b5_prithvi partA && .venv-ml/bin/python -m experiments.b5_report
.venv-ml/bin/python -m experiments.b5_export --model unet
python -m experiments.b5_exam run rule && python -m experiments.b5_exam run burn-unet
python -m experiments.b5_exam report burn-unet
# yes/no thresholds (the whole benchmark once with today's path, then the fits)
python -m experiments.b6_exam && python -m experiments.b6_fit
# every clear date (a per-date copy of the burnt-land patches, about 210 GB; then U-TAE and its control)
.venv-ml/bin/python -m core.data.impactmesh_ts
.venv-ml/bin/python -m experiments.b7_train --all --slots 1,2,4,5,5,5 --batch 4 --accum 2
.venv-ml/bin/python -m experiments.b7_report
# fires of 2026: CEMS maps, patches and labels; image copies; radar and elevation; TerraMind-base-Fire's inputs
python -m core.data.cems list && python -m core.data.cems download && python -m core.data.cems patches
BURN_SOURCE=cems2026 .venv-ml/bin/python -m core.data.impactmesh_es && BURN_SOURCE=cems2026 .venv-ml/bin/python -m core.data.impactmesh_ts
python -m core.data.burn_radar && BURN_SOURCE=cems2026 python -m core.data.burn_radar
python -m core.data.tmfire_inputs build && .venv-fire/bin/python -m core.data.tmfire_inputs write
BURN_SOURCE=cems2026 python -m core.data.tmfire_inputs build && BURN_SOURCE=cems2026 .venv-fire/bin/python -m core.data.tmfire_inputs write
.venv-ml/bin/python -m experiments.b8_train --all --slots 1,2,4
python -m experiments.b8_tmfire run heldout && python -m experiments.b8_tmfire check && python -m experiments.b8_tmfire run cems2026
for m in unet utae utae_rd; do .venv-ml/bin/python -m experiments.b8_predict $m; done
.venv-ml/bin/python -m experiments.b8_report
.venv-ml/bin/python -m experiments.b8_export
python -m experiments.b8_exam run burn-unet && python -m experiments.b8_exam run burn-utae && python -m experiments.b8_exam report
# three ideas from an independent review: flooded land (1), SpaceNet 7 (2), MMFlood (3)
python -m experiments.b9_flood_data partc && python -m experiments.b9_flood_data chips
.venv-ml/bin/python -m experiments.b9_report ref && .venv-ml/bin/python -m experiments.b9_train --all --gpus 1,2,4,5
python -m experiments.b9_report
python -m core.data.spacenet7 labels && python -m core.data.spacenet7 && python -m experiments.b9_sn7_inputs
.venv-ml/bin/python -m experiments.b9_train --all --gpus 1,2,4,5 --model s,sn && python -m experiments.b9_report2
python -m core.data.mmflood_s1rtc && .venv-ml/bin/python -m experiments.b9_radar_train prep
.venv-ml/bin/python -m experiments.b9_radar_train --all --gpus 1,2,4,5 && .venv-ml/bin/python -m experiments.b9_report3
python -m experiments.b10_fire_dates questions && for c in A S T; do python -m experiments.b10_fire_dates run $c; done
python -m experiments.b10_fire_dates report
python -m experiments.b4_exam run terramind+water B && python -m experiments.b4_exam run terramind+water C
```
