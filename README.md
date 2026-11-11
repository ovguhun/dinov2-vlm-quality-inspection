# DINOv2-Based Industrial Anomaly Detection with Vision-Language-Model-Generated Quality Control Reports

Code and results for the MSc thesis in Data Science of Mehmet Övgü Hun,
University of Debrecen, Faculty of Informatics. The study evaluates Dinomaly,
a DINOv2-based anomaly detector, alongside PatchCore and EfficientAD-S on
MVTec AD and VisA. It then tests whether a frozen Qwen2.5-VL-3B-Instruct
writes more accurate inspection reports when it receives Dinomaly's evidence
and a retrieved knowledge card about the product category.

## Overview

Qwen writes a structured inspection report for each of 54 test images, one
normal and one anomalous from each of the 27 categories, in three conditions:

| Condition | Input to Qwen |
| --- | --- |
| I0: image | the test image |
| D0: + detector | the image with Dinomaly's score, cutoff and prediction, and an overlay of its anomaly map when the prediction is anomalous |
| D1: + knowledge card | D0 with a knowledge card about the product category, when E5-small-v2 retrieves that category's own card |

Research questions, as stated in the thesis (VLM: vision-language model):

- **RQ1:** How does anomaly detection based on patch features from the
  frozen pretrained image encoder DINOv2 (implemented via Dinomaly) perform
  in terms of image-level classification and pixel-level localisation when
  compared to PatchCore and EfficientAD-S on the MVTec AD and VisA datasets?
- **RQ2:** When a VLM is tasked with generating quality control reports
  about inspected products, how does providing anomaly scores and
  anomaly-map visualisations from a DINOv2-based anomaly detector affect the
  correctness of the resulting reports and their evidence consistency in
  comparison to using product images alone?
- **RQ3:** When generating inspection reports conditioned on information
  from an anomaly detector, how does supplementing the VLM with potentially
  relevant information retrieved from a curated defect knowledge base affect
  the correctness of the resulting reports and the share of reports that
  contain unsupported or incorrect statements?

Key findings:

- Dinomaly has the highest mean image AUROC and AU-PRO on both datasets.
  Mean pixel AUROC differs by at most 0.4 points between the detectors.
- With Dinomaly's evidence (D0), correct normal/anomalous status rises from
  34 to 42 of 54 reports, and reports with a material error fall from 32
  to 26.
- The knowledge card (D1) leaves correct status at 42 of 54; useful reports
  go from 24 to 25, and reports with a material error from 26 to 28.

## Results

Both studies at a glance; click a chart for its full size.

<table>
  <tr>
    <td width="50%" align="center" valign="middle"><a href="results/detector/dataset_means.png"><img src="results/detector/dataset_means.png" alt="Mean performance of PatchCore, EfficientAD-S and Dinomaly on MVTec AD and VisA, seven metrics in per cent"></a></td>
    <td width="50%" align="center" valign="middle"><a href="results/reports/status_confusion_matrices.png"><img src="results/reports/status_confusion_matrices.png" alt="Report status against the image label for I0, D0, D1 and Dinomaly alone, on the same 54 images"></a></td>
  </tr>
  <tr>
    <td align="center"><sub><b>Detectors (RQ1).</b> Mean over the product categories of each dataset, in per cent. I and P mark image-level and pixel-level metrics.</sub></td>
    <td align="center"><sub><b>Reports (RQ2, RQ3).</b> Report status against the image label on the same 54 images. Each cell gives the count and its share of the 27 images in that row; “uncertain” never counts as correct.</sub></td>
  </tr>
</table>

### Anomaly detection (RQ1)

Mean over the categories of each dataset, in per cent; the best value per
dataset is in bold:

| Detector | Dataset | Image AUROC | Pixel AUROC | AU-PRO |
| --- | --- | ---: | ---: | ---: |
| PatchCore | MVTec AD | 99.09 | 98.13 | 93.16 |
| EfficientAD-S | MVTec AD | 99.04 | **98.43** | 94.46 |
| Dinomaly | MVTec AD | **99.69** | 98.35 | **94.89** |
| PatchCore | VisA | 93.35 | 98.37 | 88.57 |
| EfficientAD-S | VisA | 97.91 | 98.77 | 91.90 |
| Dinomaly | VisA | **98.85** | 98.77 | **95.10** |

On VisA, EfficientAD-S and Dinomaly both round to 98.77 in pixel AUROC, so
neither value is in bold.
Image metrics use each detector's own image score. Pixel metrics and AU-PRO
(up to a false-positive rate of 0.3) are computed on a 256 x 256 grid over
the image region that all three anomaly maps cover, so they are not
comparable with the values in the detector papers. Per-category results,
image and pixel AP and F1-max are in each detector's `metrics.csv` and
`summary.csv` under [results/detector/](results/detector/).

Image AUROC, AU-PRO, pixel AUROC and pixel AP for every category are drawn in
[comparison_mvtec_ad.png](results/detector/comparison_mvtec_ad.png) and
[comparison_visa.png](results/detector/comparison_visa.png).

### Inspection reports (RQ2, RQ3)

Reports out of 54 per condition ([results/reports/](results/reports/)):

| | I0: image | D0: + detector | D1: + knowledge card |
| --- | ---: | ---: | ---: |
| Correct normal/anomalous status | 34/54 (63.0%) | 42/54 (77.8%) | 42/54 (77.8%) |
| Useful report | 21/54 (38.9%) | 24/54 (44.4%) | 25/54 (46.3%) |
| Report with a material error | 32/54 (59.3%) | 26/54 (48.1%) | 28/54 (51.9%) |

Dinomaly's own prediction, anomalous when its score is above a per-category
cutoff (the 0.98 quantile of its scores on the category's other normal test
images), is correct for 53 of the 54 images. In D1, 24 of the 27 categories
retrieved their own knowledge card; fryum, macaroni1 and pcb1 retrieved
another category's, so their D1 reports got no card.

Each report is graded against reference answers for its image. A material
error is an explicit statement or recommendation that is contradicted or
lacks necessary support. The material-error grade is a yes or no per report
and does not distinguish unsupported from contradicted statements. The share
of reports with a material error is the second outcome named in RQ3.
[results/reports/README.md](results/reports/README.md) describes the grades,
and `summary.csv` and `paired_changes.csv` there hold all measures.

## Repository layout

| Folder | Contents |
| --- | --- |
| `scripts/` | the eight pipeline steps, run in order (see below) |
| `src/` | the modules the steps share: dataset reading, metrics, prediction files, figures |
| `configs/` | reporting settings, the prompt and the report schema |
| `knowledge/` | the 27 knowledge cards, one per category, with sources |
| `third_party/` | unmodified PatchCore, EfficientAD and Dinomaly code |
| `results/` | the reported results |
| `tests/` | unit tests |
| `licenses/` | licence texts for other included code |

## Installation

Linux with Python 3.12 and an NVIDIA GPU that supports CUDA 13.0 and
bfloat16. The reported runs used an RTX 5070 Laptop GPU.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover tests
```

Two of the tests read the datasets and are skipped until they are in place
(see [Data and weights](#data-and-weights)).

## Quick check

The report summaries can be reproduced in seconds, without data, weights or
a GPU:

```bash
mkdir -p outputs/reports
cp results/reports/{cases,reports,grades}.csv outputs/reports/
python -m scripts.08_evaluate_reports
cmp outputs/reports/summary.csv results/reports/summary.csv
cmp outputs/reports/paired_changes.csv results/reports/paired_changes.csv
```

`cmp` prints nothing when the files are identical. Delete `outputs/reports/`
afterwards: 07 does not overwrite an existing `reports.csv`, and 08 accepts a
grade only for the answer it was given for.

## Data and weights

Extract the datasets so that these paths exist:

| Dataset | Path | Source |
| --- | --- | --- |
| MVTec AD (CC BY-NC-SA 4.0) | `data/MVTec-AD/<category>/` | [mvtec.com](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) |
| VisA (CC BY 4.0) | `data/VisA/<category>/` | [VisA_20220922.tar](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar) |
| VisA one-class split | `data/VisA/split_csv/1cls.csv` | in the VisA archive; also in [amazon-science/spot-diff](https://github.com/amazon-science/spot-diff/blob/2a692ab575001cbde74d402d897a7286086c6199/split_csv/1cls.csv) |
| Imagenette, full size (only for 02) | `data/imagenette2/train/` | [imagenette2.tgz](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz) |

Then run `python -m unittest discover tests` again: the two dataset tests now
check both datasets' image counts against the official numbers.

Save the weights under these paths:

| Weights | Path | Source |
| --- | --- | --- |
| EfficientAD-S teacher | `weights/efficientad/teacher_small.pth` | [`models/teacher_small.pth`](https://github.com/nelson1425/EfficientAD/blob/fcab5146f84ae17597044ad5ddf1656ccf805401/models/teacher_small.pth) in nelson1425/EfficientAD |
| Dinomaly-B, MVTec AD | `weights/dinomaly/mvtec_vitb_it20k.pth` | MVTec-AD row of [Trained model weights](https://github.com/guojiajeremy/Dinomaly/blob/1f252be03a918789b19848f0ca37166e7c28dada/README.md#trained-model-weights) in the Dinomaly README |
| Dinomaly-B, VisA | `weights/dinomaly/visa_vitb_it20k.pth` | VisA row of the same table |

To check them, run this from the repository root:

```bash
sha256sum -c <<'EOF'
b98a6a139087a0d743783ee59f367f0284af6003eb1a5fed0ce92bd9c0acd337  weights/efficientad/teacher_small.pth
2108f25729b303938306ddafdf76e8275f8654b106f044b45e2a5bec9d29085c  weights/dinomaly/mvtec_vitb_it20k.pth
e5031cb3f69f9bea75fa429359295d1599b09eef87b684d601c91d1d71bc959e  weights/dinomaly/visa_vitb_it20k.pth
EOF
```

The other models download on first use: torchvision's WideResNet-50-2
(`IMAGENET1K_V1` weights) for PatchCore, and E5-small-v2 and
Qwen2.5-VL-3B-Instruct at the revisions in `configs/reporting.json`.

## Reproducing the experiments

Run each script from the repository root as `python -m scripts.<name>`, for
example `python -m scripts.03_dinomaly`. For the full study, run all eight in
this order; for the reporting study alone, 03 and then 05 to 08 are enough.
Times refer to the RTX 5070 Laptop GPU.

| Script | What it does | Time |
| --- | --- | --- |
| `01_patchcore.py` | PatchCore scores and anomaly maps | about 1.5 h |
| `02_efficientad.py` | trains EfficientAD-S per category; scores and maps | about 48 h |
| `03_dinomaly.py` | Dinomaly scores and maps from the released checkpoints | about 5 min |
| `04_compare_detectors.py` | RQ1 metrics on the shared image region | about 10 min |
| `05_select_report_images.py` | the 54 report images and the per-category cutoffs | seconds |
| `06_retrieve_knowledge.py` | one knowledge card per category (CPU) | under a minute |
| `07_generate_reports.py` | the 162 reports with Qwen2.5-VL-3B-Instruct | about 9 min, plus loading the model |
| `08_evaluate_reports.py` | the RQ2 and RQ3 summaries from the grades | seconds |

02 trains the 27 categories one after another and cannot resume: if it stops,
the next run starts again from the first category.

05 draws the report images at random (seed 0) from the test images that are
not listed under `excluded_images` in `configs/reporting.json`. Those 117
images were used for trial runs while the prompt and the report settings were
prepared; leaving them out keeps the 54 report images separate from those
runs.

07 stops if `outputs/reports/reports.csv` already exists, so a finished run
is never overwritten. On its first run, 08 writes an empty grading sheet,
`outputs/reports/grades.csv`, and stops: grade the reports as described in
[results/reports/README.md](results/reports/README.md) and run it again.

## Reproducibility

`results/` holds the reported results and is never written by the scripts;
they write to `outputs/`, under the same paths:

| In `outputs/` and `results/` | |
| --- | --- |
| `detector/<detector>/metrics.csv`, `summary.csv` | RQ1 metrics |
| `detector/<detector>/<dataset>/<category>.png` | example images; `results/` keeps four selected figures per detector, in a different layout |
| `retrieval/retrieved_cards.csv` | the knowledge card for each category |
| `reports/cases.csv`, `reports.csv`, `grades.csv`, `summary.csv`, `paired_changes.csv` | the reporting study; `results/` also has `references.csv` |

Three kinds of chart have no counterpart in `outputs/`, because no script
here draws them: `results/detector/dataset_means.png` and
`comparison_<dataset>.png`, drawn from the three detectors' `summary.csv` and
`metrics.csv` files, and `results/reports/status_confusion_matrices.png`,
drawn from the statuses in `cases.csv` and `reports.csv`.

Expected agreement with the reported results:

- **Dinomaly** (03): with the pinned software on the GPU used for the
  reported runs, scores and maps are bit-identical to the reported ones; on
  other GPUs the last digits can change.
- **PatchCore and EfficientAD-S** (01, 02) are trained anew, so their
  numbers should come close to the reported ones without necessarily being
  equal. EfficientAD-S training on a GPU is not bit-reproducible, and the
  reported models were trained with Python 3.11 and PyTorch 2.13.
- **Retrieval** (06) should select the same cards.
- **Reports** (07): the D0 and D1 prompts quote Dinomaly's score in full, so
  regenerated reports can match the reported ones only when `cases.csv`
  does; the GPU and library versions matter as well. The grades in
  `results/reports/` belong to the reports there, and together they give the
  published summaries (see the [quick check](#quick-check)).

## Licences

The code is under the Apache License 2.0 ([LICENSE](LICENSE)). Code and text
from other projects keep their licences: see [NOTICE](NOTICE),
[third_party/](third_party/) and [licenses/](licenses/). The figures in
`results/detector/<detector>/` show dataset images and are shared under the
dataset licences (see NOTICE). MVTec AD is under CC BY-NC-SA 4.0
(non-commercial), VisA under CC BY 4.0, E5-small-v2 under the MIT License and
Qwen2.5-VL-3B-Instruct under the Qwen Research License.

## Citation

[CITATION.cff](CITATION.cff) gives the citation for this repository (GitHub
shows it under "Cite this repository") and lists the works it builds on.
