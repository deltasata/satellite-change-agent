# Methods

Every learned component in this system had to beat the rule it replaces, under a protocol written down
**before any score was computed**. This page gives those protocols, the data, and the full results.

- [1. Principle: a model replaces a rule only if it wins](#1-principle-a-model-replaces-a-rule-only-if-it-wins)
- [2. Data](#2-data)
- [3. Water in optical images](#3-water-in-optical-images)
- [4. Water in radar images](#4-water-in-radar-images)
- [5. The answer path, end to end](#5-the-answer-path-end-to-end)
- [6. Comparison with published Sen1Floods11 results](#6-comparison-with-published-sen1floods11-results)
- [7. The agent benchmark](#7-the-agent-benchmark)
- [8. Limits](#8-limits)

## 1. Principle: a model replaces a rule only if it wins

1. **Test data fixed in advance, used once.** No training, setting or threshold is chosen on them.
2. **One main number per step, chosen in advance.** For pixel maps: the IoU (intersection over union)
   of the class that matters, pooled over all test pixels: IoU = ΣTP / (ΣTP + ΣFP + ΣFN). For the answer
   path: accuracy of the final yes / no / cannot-tell answer.
3. **Held-out events, not held-out tiles.** Tiles of one flood share weather, sensor geometry, soil and
   water colour, so they are not independent. Every model is trained with **one whole flood event left
   out** and tested on that event, for each event in turn.
4. **Paired comparison with an honest interval.** Rule and model are scored on the same pixels. The 95 %
   interval of the difference comes from a **cluster bootstrap over flood events** (10 000 resamples of
   whole events). **The model replaces the rule only if the whole interval lies above zero.**
5. **Then the whole answer path** is rerun with the model in place; the answers must not get worse.
6. **Every result is reported, including the losses.**

## 2. Data

| data | use | access |
|---|---|---|
| Sentinel-2 Level-2A (surface reflectance) | the pipeline's optical images | Element84 Earth Search, free COGs |
| Sentinel-1 RTC (radiometrically terrain-corrected backscatter) | the pipeline's radar images | Microsoft Planetary Computer, free, no account |
| Sen1Floods11 (Bonafilia et al. 2020) | 446 hand-labelled 512 × 512 tiles from 11 flood events, with Sentinel-1 and Sentinel-2 | public bucket `gs://sen1floods11` |

**The domain-shift trap, and how it is avoided.** Sen1Floods11's own images are Level-1C optical (top of
atmosphere) and Earth-Engine sigma-naught radar. The pipeline reads Level-2A optical and RTC gamma-naught
radar. A model that is excellent on the data set's images can fail on the pipeline's: the published
Prithvi-EO-2.0 Sen1Floods11 model scores **water IoU 0.817 on Level-1C but 0.634 on Level-2A** (official
test and Bolivia tiles), below the simple rule NDWI > 0 on the same Level-2A tiles (0.638). So two copies of Sen1Floods11 were built **in the pipeline's own
image formats**, same scene, read onto the hand label's pixel grid:

- `core/data/sen1floods11_l2a.py`: Level-2A, 376 tiles, 10 events (India 2016 predates the L2A archive).
  Pixel correlation with the original Level-1C: median 0.989.
- `core/data/sen1floods11_s1rtc.py`: Sentinel-1 RTC, 379 tiles, 10 events (Paraguay's scene is not in
  the archive). VH correlation with the original after a 9 × 9 pixel average: 0.90-0.98 per event; RTC
  is 1.1-2.1 dB brighter, as expected for gamma naught.

All decisions are made on these copies.

## 3. Water in optical images

**Question.** Does a learned model mark water in a Sentinel-2 L2A image better than NDWI > 0
(NDWI = (green − NIR) / (green + NIR))?

**Detectors.**
- Rules: NDWI > 0; NDWI > t and MNDWI > t (MNDWI uses short-wave infrared), t fitted on the training events.
- Prithvi-EO-2.0-300M-TL (NASA / IBM) with a UperNet decoder, fine-tuned (the published flood setup).
- TerraMind-1.0-base (IBM / ESA) with a UperNet decoder, fine-tuned.
- A U-Net with a ResNet-18 encoder, trained from scratch.

**Training, fixed in advance and identical for all:** 6 bands (blue, green, red, B8A, B11, B12); random
224 × 224 crops, flips and 90° rotations; batch 16; AdamW, weight decay 0.05; learning rate 5e-5
(pretrained) or 1e-3 (U-Net), cosine schedule; 50 epochs (U-Net 100); cross-entropy with unlabelled and
cloudy pixels ignored; bf16. Prediction on full tiles from overlapping 224 windows. No early stopping,
so no test event is touched. The pretrained encoders were verified tensor-by-tensor against the
published weights.

**Results** (L2A, 376 tiles, 10 events, each held out in turn):

| detector | water IoU | minus NDWI > 0 [95 % interval] |
|---|---|---|
| NDWI > 0 | 0.690 | |
| NDWI > t, fitted | 0.621 | −0.069 [−0.232, +0.038] |
| MNDWI > t, fitted (best rule) | 0.777 | +0.087 [+0.044, +0.162] |
| **U-Net, from scratch** | **0.833** | **+0.143 [+0.083, +0.219]** |
| Prithvi-EO-2.0-300M, fine-tuned | 0.712 | +0.022 [−0.173, +0.187] |
| TerraMind-1.0-base, fine-tuned | 0.698 | +0.008 [−0.184, +0.167] |

The U-Net is adopted. It also beats the best rule (MNDWI fitted) by +0.056 [+0.025, +0.092].

| flood | NDWI > 0 | MNDWI fitted | U-Net | Prithvi | TerraMind |
|---|---|---|---|---|---|
| Bolivia | 0.579 | 0.644 | **0.801** | 0.787 | 0.767 |
| Ghana | 0.675 | 0.765 | **0.778** | 0.770 | 0.763 |
| Mekong | 0.892 | 0.909 | **0.924** | 0.906 | 0.896 |
| Nigeria | 0.865 | 0.897 | **0.925** | 0.919 | 0.907 |
| Pakistan | 0.410 | **0.677** | 0.618 | 0.129 | 0.119 |
| Paraguay | 0.566 | 0.616 | **0.757** | 0.744 | 0.719 |
| Somalia | 0.491 | 0.697 | **0.763** | 0.751 | 0.739 |
| Spain | 0.648 | 0.741 | 0.831 | **0.832** | 0.810 |
| Sri Lanka | 0.759 | 0.811 | 0.850 | **0.851** | 0.831 |
| USA | 0.449 | 0.747 | **0.781** | 0.766 | 0.742 |

**Findings.**
- **Muddy water defeats NDWI.** Suspended sediment reflects near infrared, so flood water can have
  NDWI < 0 (USA tile 217598: median −0.15; the rule finds 2 % of the water, the U-Net 82 %). Water stays
  dark in short-wave infrared, which the U-Net (and MNDWI) exploit.
- **The foundation models are within ~0.03 of the U-Net on 9 of 10 events, and collapse on one.**
  Pakistan is a hazy scene whose dry land is brighter in blue (0.13) than any training event's
  (0.04-0.08); both foundation models label large dry areas as water (IoU 0.13 and 0.12). This matches
  the PANGAEA benchmark (Marsocci et al. 2024), where a U-Net also tops Sen1Floods11.

![example](figures/b1_USA_217598.png)

## 4. Water in radar images

**Question.** Radar sees through cloud. Which detector marks water best in Sentinel-1 RTC: a backscatter
threshold or a learned model?

**Detectors.**
- Rules: VH < t and VV < t, t fitted on the training events (0.1 dB grid); Otsu's threshold per tile on
  VH (the Sen1Floods11 paper's baseline, nothing fitted).
- U-Net (ResNet-18), from scratch, input VV and VH in dB.
- TerraMind-1.0-base, modality S1RTC (TerraMind was pretrained on RTC radar), UperNet decoder, fine-tuned.

Training as in section 3. **New: 3 seeds per model and held-out event**; the main number uses the mean
of the 3 seeds' probabilities.

**Results** (RTC, 379 tiles, 10 events, each held out in turn):

| detector | water IoU | minus VH < t [95 % interval] | single seeds |
|---|---|---|---|
| VH < t, fitted (t ≈ −20.5 dB) | 0.477 | | |
| VV < t, fitted | 0.427 | −0.050 [−0.162, +0.036] | |
| Otsu per tile, VH | 0.199 | −0.278 [−0.351, −0.187] | |
| U-Net, from scratch | 0.558 | +0.081 [−0.002, +0.150] | 0.530, 0.547, 0.578 |
| **TerraMind-1.0-base, fine-tuned** | **0.611** | **+0.134 [+0.090, +0.192]** | 0.606, 0.598, 0.606 |

TerraMind minus U-Net: +0.053 [+0.008, +0.123]. **TerraMind is adopted for radar.** In contrast to the
optical case, the pretrained foundation model wins clearly on radar, and its seeds agree closely (spread
0.008 against 0.048 for the U-Net): radar water is a harder, speckled signal where pretraining pays.

| flood | VH fitted | U-Net | TerraMind |
|---|---|---|---|
| Bolivia | 0.568 | 0.618 | **0.641** |
| Ghana | 0.382 | 0.398 | **0.534** |
| India | 0.400 | **0.479** | 0.468 |
| Mekong | 0.716 | **0.819** | 0.804 |
| Nigeria | 0.563 | 0.746 | **0.799** |
| Pakistan | 0.199 | 0.310 | **0.422** |
| Somalia | 0.125 | **0.261** | 0.250 |
| Spain | 0.607 | 0.682 | **0.704** |
| Sri Lanka | 0.599 | 0.420 | **0.667** |
| USA | 0.419 | 0.467 | **0.498** |

![radar example](figures/b2_Ghana_868803.png)

## 5. The answer path, end to end

The benchmark's flood questions ("Was this 5 km area flooded around date D, compared with the weeks
before?") are built from Sen1Floods11 hand labels (yes: tiles with ≥ 15 % flood water; no: dry tiles).
Held fixed: the true periods (before: D−60 to D−10; after: D−3 to D+3), the images, the cloud mask, the
decision rule (new water ≥ 5 % of the area → yes). Only the water detector changes. Each question is
answered by models that never saw its flood event.

**Optical images only** (78 questions): NDWI > 0 0.667 → U-Net 0.744 (+0.077 [+0.000, +0.151]). The
U-Net changes 6 answers, all from a wrong "no" to a right "yes", and breaks none.

**Radar where the optical images cannot tell.** Clouds leave 19 of the 72 questions outside India
without a usable optical image. For those, the pipeline now reads every Sentinel-1 scene of the same
periods; TerraMind gives a water probability per scene; per period, the per-pixel median; water where
> 0.5; the same decision rule.

| answer from | accuracy (72 questions) | "cannot tell" |
|---|---|---|
| optical, NDWI > 0 | 0.639 | 19 |
| optical, U-Net | 0.722 | 19 |
| optical U-Net, then radar with the VH rule | 0.875 | 5 |
| **optical U-Net, then radar with TerraMind** | **0.889** | **5** |

Gain over optical alone: **+0.167 [+0.031, +0.346]**. Radar answered 14 previously unanswerable
questions, 12 of them correctly. The pipeline's own code (`measure_change`) gives the experiment's answer on all 78 questions. The 6 India 2016 questions have no optical image at all; with radar, 2 of 6 are answered
correctly (the 2 "no" questions; the 4 "yes" questions were answered "no").

## 6. Comparison with published Sen1Floods11 results

Here the setup is the published one: the data set's own images (radar: Earth Engine sigma naught;
optical: Level-1C, 13 bands) and its official split (train 252, validation 89, test 90 tiles; Bolivia,
15 tiles, held out apart). Settings were chosen on the validation part only: for the U-Net, encoder
(ResNet-18 / 34) and loss (cross-entropy, or cross-entropy + Dice), 4 combinations; for TerraMind, the
learning rate (5e-5 / 1e-4). The chosen setting was trained with 5 seeds; whether to average the 5 seeds
and to average 8 flipped and rotated views at prediction time was also decided on validation. The test
part was scored once, at the end. Pooled over all test pixels, as in the published papers.

| input | model | test water IoU | test mIoU | single seeds (water IoU) | published (same test part) |
|---|---|---|---|---|---|
| radar | rule VH < t (fitted) | 0.533 | 0.728 | | |
| radar | U-Net | 0.663 | 0.806 | 0.650-0.666 | |
| radar | **TerraMind-1.0-base** | **0.672** | **0.811** | 0.654-0.679 | TerraMind-1.0-base: 0.680 / 0.811 [1] |
| optical | **U-Net** | **0.830** | **0.903** | 0.822-0.842 | TerraMind-1.0-base: 0.823 / 0.897 [1] |
| optical | TerraMind-1.0-base | 0.824 | 0.899 | 0.812-0.824 | |
| radar + optical | **U-Net** | **0.844** | **0.911** | 0.833-0.844 | U-Net: mIoU 0.914 [2] |
| radar + optical | TerraMind-1.0-base | 0.821 | 0.897 | 0.801-0.824 | |

[1] Jakubik et al. 2025, Table 8. [2] PANGAEA, Marsocci et al. 2024, Table 5.

**Our numbers reproduce the published ones to within 0.01**, in both directions: optical U-Net +0.007
water IoU over the published TerraMind result; radar TerraMind −0.008; radar + optical U-Net −0.003 mIoU.
These differences are smaller than the spread between single seeds (about 0.02), so the honest
statement is *on par with the published state of the art*, not *better*. Two things this establishes:
our training setup is sound, and the large losses in sections 3 and 4 come from the harder test (a whole
flood never seen, in the pipeline's own image formats), not from weak training. Combining radar and
optical gives the best water IoU of all (0.844). On Bolivia, the held-out country, the optical models
keep their accuracy (0.79-0.81) while the radar models drop (0.49).

## 7. The agent benchmark

`eval/benchmark.jsonl` (44 questions) and `eval/benchmark_v2.jsonl` (150): flood questions from
Sen1Floods11 labels, construction and wildfire sites with verified dates, stable control sites, and
questions designed to be unanswerable (polar night, before Sentinel-2, the future). Three orchestrations
were compared over one MCP tool server: a free agent loop (Claude Agent SDK), a fixed LangGraph workflow
with a local model (Qwen3, vLLM, one GPU), and the same graph with Claude.

![agents](figures/results_variants.png)

- **Model, orchestration fixed:** Claude vs Qwen3-8B in the same graph: 19 questions fixed, 2 broken
  (sign test p = 0.0002).
- **Arithmetic belongs in code:** computing the date windows in code instead of the model raised
  Qwen3-8B from 0.48 to 0.66.
- **Reasoning mode:** Qwen3-8B with thinking on: 0.48 → 0.77 (p = 0.002), at 5× the latency.

The final design follows from this: Claude holds the conversation, **all measurement is one
deterministic function** (`measure_change`), and two checks in code verify that the reply's answer and
every number in it come from that function's results.

## 8. Limits

- 10 flood events give wide intervals; the radar U-Net's gain is not significant (interval touches 0).
- One seed per model in the optical comparison; three in the radar comparison.
- The foundation models used the published Prithvi training settings, not tuned for each model.
- The benchmark's flood questions come from the same 10-11 events as the training data (each question is
  answered by a model that never saw its event, but they are not new places).
- Radar answers have no evidence figure yet.
