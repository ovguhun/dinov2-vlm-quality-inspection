# Report results

The inspection reports of the thesis and their grades. `cases.csv` comes from
`scripts/05_select_report_images.py`, `reports.csv` from
`scripts/07_generate_reports.py` and the two summaries from
`scripts/08_evaluate_reports.py`. The references and grades are the thesis
assessment, which the thesis describes.

| File | Contents |
| --- | --- |
| `cases.csv` | The 54 test images, one normal and one anomalous image from each of the 27 categories, with the label and Dinomaly's score, cutoff and prediction. |
| `references.csv` | What a correct report says about each image: defect type, location, appearance impact (`severity_estimate`), visible evidence and a reasonable next step. Normal images have no defect type, location or impact; `unavailable` explains the answers that could not be settled. |
| `reports.csv` | Qwen's 162 reports, one per image and condition (I0, D0, D1). |
| `grades.csv` | The grades of each report, explained below. `answer_digest` is a short fingerprint of the report's raw answer, and `review_id` is the report's number in the thesis. |
| `summary.csv` | Every measure per condition. The `detector` column scores Dinomaly's own predictions. |
| `status_confusion_matrices.png` | The report status against the image label, per condition and for Dinomaly alone, drawn from the statuses in `cases.csv` and `reports.csv`. |
| `paired_changes.csv` | On the same image, how often the report got better, stayed the same or got worse from I0 to D0 and from D0 to D1. "card shown" keeps the 48 images whose D1 report got its category's knowledge card. `exact_p` is a two-sided exact sign test of better against worse on the images that changed. |

The inspection status needs no grade: 08 compares it with the label, and
"uncertain" never counts as correct.

## Grades

Each report is read next to its original image and the image's reference
answers.

**usefulness.** Length, fluency and citations earn nothing by themselves.

- `useful`: enough supported, case-specific help with the finding, its
  evidence, its uncertainty and the next step, and nothing material to
  correct.
- `partly useful`: some supported help that can be used on its own, but
  omissions, vagueness, distraction or mixed content need clarifying,
  completing or correcting.
- `not useful`: no usable case-specific help, or a materially misleading
  finding, site, evidence or advice.

**material_error.** `yes` when at least one explicit statement or
recommendation is contradicted or lacks necessary support, for example a
wrong normal/anomalous status, a wrong appearance-impact band on a real
defect or unsupported intervention advice. Missing or vague content alone is
not an error. One `useful` report (review-155) has a material error in its
advice.

**contradiction.** `yes` when two statements disagree about the same feature,
scope and certainty, judged from the wording alone, not from the label. A
normal status beside an unqualified diagnosis of damage, or advice that
presumes it, is a contradiction; a neutral observation, a filled-in field or
conditional inspection advice alone is not. A coherent but wrong report has no
contradiction.

**location_grade.** The location the report gives, on the full image, with
left and right as the viewer sees them. On anomalous images, against the
reference location:

- `correct`: the relevant sites and their extent, with no wrong site;
- `partial`: the right site, but coarse, incomplete or mixed with an
  unsupported addition;
- `wrong`: a specific but wrong site, or an explicit "none";
- `not established`: a location without useful specificity;
- `missing`: no location, either null or a field the report lacks;
- `reference unavailable`: the reference cannot place every annotated site
  (case-040), so the location is not graded.

On normal images: `appropriate n/a` when the location is null, `missing`
when the report lacks the field, `wrong` when the report places a defect that
is not there, and `not established` for a vague location or an ordinary
feature. Only an invalid report can lack a field; in `reports.csv` that cell
is empty like a null, and `raw_output` shows which it is.

**detector_score, detector_decision, detector_attribution, detector_overlay.**
Whether a D0 or D1 report reads the detector evidence it was given faithfully:

- score: keeps the score's value and meaning; a raw score is not a
  probability;
- decision: keeps the cutoff, the rule "anomalous when score > cutoff", the
  prediction and that it may be wrong;
- attribution: keeps apart what the detector shows, what the image shows and
  general knowledge; an anomaly map alone establishes no defect type, cause or
  impact;
- overlay: keeps where the overlay is red, that red is only relative within
  the image, and which region was inspected.

The grades are `fully faithful`, `partial` (mixed or incomplete),
`inconsistent` (misread), `unattempted` (the report does not interpret it) and
`n/a` (not given: everything in I0, and the overlay when none was shown).
Quoting the report's own verdict or giving an ordinary location is not
interpreting the detector, and disagreeing with the detector is not a
misreading.

## Worked example

case-008 in D0, a carpet with metal contamination
(`mvtec_ad/carpet/test/metal_contamination/000.png`). Dinomaly predicted
anomalous, so the report was given the overlay.

| Field | Reference | Report |
| --- | --- | --- |
| inspection_status | anomalous (label) | anomalous |
| defect_type | Metal contamination / foreign fragment | dark spot |
| location | Slightly left of centre and below the middle of the carpet image | center of the carpet |
| severity_estimate | limited | limited |
| visual_evidence | A small elongated grey fragment lies across the weave slightly left of centre and below the middle; the surrounding weave remains recognizable. | A small, irregularly shaped dark spot is visible in the center of the carpet. |
| recommended_action | Inspect the fragment and adjacent fibres closely. | Conduct a closer inspection to determine the cause and extent of the defect. |

- usefulness `partly useful`: it notices the real localised feature, but does
  not identify the fragment or locate it precisely; the impact and the
  closer-inspection advice are reasonable.
- material_error `no`: "dark spot" is coarse rather than wrong, and the
  status, impact and advice are supported.
- contradiction `no`: the status, description and advice agree.
- location_grade `partial`: "center of the carpet" is near the fragment, but
  misses its position left of centre and below the middle.
- detector grades `unattempted`: the report does not mention the score, the
  cutoff or the overlay.

## Grading new reports

When `outputs/reports/grades.csv` does not exist, 08 writes it with one row
per report and empty grades, with `n/a` already in the detector columns for
evidence a report was not given. Fill it in with the words above and run 08
again; it stops with a message if a grade is missing, repeated or does not
fit its report, or if a row's `answer_digest` no longer matches the report's
answer. With this folder's `cases.csv`, `reports.csv` and
`grades.csv` in `outputs/reports/`, 08 writes the `summary.csv` and
`paired_changes.csv` found here.

The grades here belong to this folder's `reports.csv`. Reuse one only where
your report is the same, and grade the rest. Reports can only be the same
when `cases.csv` is, because the D0 and D1 texts quote the full score.
