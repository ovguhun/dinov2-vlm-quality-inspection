"""Summarise the reports written by 07.

Validity and status correctness are counted from the reports ("uncertain" is
never correct). Usefulness, material errors, contradictions, location and the
reading of detector evidence come from a grading sheet with one row per
report (see results/reports/README.md). If the sheet (files.grades in
configs/reporting.json) is missing, the script writes an empty one and stops.
It also stops when a row is missing or repeated, a grade does not fit its
report, or answer_digest shows that the grades belong to another answer.

Writes outputs/reports/summary.csv and paired_changes.csv (better, same or
worse on the same image, with an exact sign test).
Run from the repository root: python -m scripts.08_evaluate_reports
"""

import csv
import hashlib
import itertools
import json
from pathlib import Path

from scipy.stats import binomtest

CONFIG = Path("configs/reporting.json")
USEFULNESS = ("not useful", "partly useful", "useful")  # worst to best
FIDELITY = ("fully faithful", "partial", "inconsistent", "unattempted", "n/a")
DETECTOR_GRADES = (
    "detector_score",
    "detector_decision",
    "detector_attribution",
    "detector_overlay",
)
GRADES = {
    "usefulness": USEFULNESS,
    "material_error": ("yes", "no"),
    "contradiction": ("yes", "no"),
    "location_grade": {  # by the image's label
        "anomalous": (
            "correct",
            "partial",
            "wrong",
            "not established",
            "missing",
            "reference unavailable",
        ),
        "normal": ("appropriate n/a", "wrong", "not established", "missing"),
    },
    **dict.fromkeys(DETECTOR_GRADES, FIDELITY),
}
# Each paired measure as a value that is higher for the better report.
PAIRED = {
    "status": lambda row: row["inspection_status"] == row["label"],
    "usefulness": lambda row: USEFULNESS.index(row["usefulness"]),
    "material error": lambda row: row["material_error"] == "no",
    "internal contradiction": lambda row: row["contradiction"] == "no",
}


def read_csv(path):
    # utf-8-sig also reads sheets that a spreadsheet saved with a BOM.
    with open(path, newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def keys(rows):
    """The sorted (case_id, condition) pairs of some rows."""
    return sorted((row["case_id"], row["condition"]) for row in rows)


def given(report, column):
    """Whether the report was given the detector evidence graded in column."""
    if column == "detector_overlay":
        return report["overlay_shown"] == "yes"
    return bool(report["detector_prediction"])


def answer_digest(report):
    """Short fingerprint of the raw answer; ties a row of grades to its answer."""
    answer = report["raw_output"].encode("utf-8")
    return hashlib.sha256(answer).hexdigest()[:12]


def blank_sheet(reports):
    """One row of empty grades per report; the detector grades are n/a
    where the report was not given that evidence."""
    return [
        {
            "case_id": report["case_id"],
            "condition": report["condition"],
            "answer_digest": answer_digest(report),
            **dict.fromkeys(GRADES, ""),
            **{
                column: "" if given(report, column) else "n/a"
                for column in DETECTOR_GRADES
            },
        }
        for report in reports
    ]


def check_grades(sheet, reports, cases):
    """The grades by (case_id, condition); stops at the first problem."""
    needed = ("case_id", "condition", "answer_digest", *GRADES)
    missing = [column for column in needed if sheet and column not in sheet[0]]
    if missing:
        raise SystemExit(f"The grading sheet lacks {', '.join(missing)}")
    if keys(sheet) != keys(reports):
        raise SystemExit("The grading sheet needs exactly one row per report")
    grades = {(row["case_id"], row["condition"]): row for row in sheet}
    for report in reports:
        key = report["case_id"], report["condition"]
        if grades[key]["answer_digest"] != answer_digest(report):
            raise SystemExit(
                f"{' '.join(key)}: the grades were given for a different answer"
            )
        for column, words in GRADES.items():
            grade = grades[key][column]
            if column == "location_grade":
                words = words[cases[report["case_id"]]["label"]]
            if grade not in words:
                raise SystemExit(
                    f"{' '.join(key)}: {column} is {grade!r}, not one of {words}"
                )
            if column in DETECTOR_GRADES:
                shown = given(report, column)
                if shown == (grade == "n/a"):
                    raise SystemExit(
                        f"{' '.join(key)}: {column} should "
                        f"{'not ' if shown else ''}be n/a"
                    )
    for case_id in cases:
        unavailable = {
            row["location_grade"] == "reference unavailable"
            for key, row in grades.items()
            if key[0] == case_id
        }
        if len(unavailable) > 1:
            raise SystemExit(
                f"{case_id}: 'reference unavailable' must be "
                "given in all of its conditions or none"
            )
    return grades


def share(count, total):
    """'count/total (percent)', or 'undefined' when total is 0."""
    if total == 0:
        return "undefined"
    return f"{count}/{total} ({100 * count / total:.1f}%)"


def rate(rows, column, value):
    """The share of the rows whose column has this value."""
    return share(sum(row[column] == value for row in rows), len(rows))


def with_label(rows, label):
    return [row for row in rows if row["label"] == label]


def status_measures(rows, status):
    """How the verdicts in column status compare with the labels."""
    normal = with_label(rows, "normal")
    anomalous = with_label(rows, "anomalous")
    correct = sum(row[status] == row["label"] for row in rows)
    return {
        "correct status": share(correct, len(rows)),
        "anomalous images called anomalous": rate(anomalous, status, "anomalous"),
        "normal images called normal": rate(normal, status, "normal"),
        "normal images called anomalous": rate(normal, status, "anomalous"),
        "anomalous images called normal": rate(anomalous, status, "normal"),
        "uncertain status": rate(rows, status, "uncertain"),
    }


def grade_measures(rows):
    """Usefulness, material errors and internal contradictions."""
    correct = [row for row in rows if row["inspection_status"] == row["label"]]
    measures = {
        level: rate(rows, "usefulness", level) for level in reversed(USEFULNESS)
    }
    measures.update(
        {
            "useful, normal images": rate(
                with_label(rows, "normal"), "usefulness", "useful"
            ),
            "useful, anomalous images": rate(
                with_label(rows, "anomalous"), "usefulness", "useful"
            ),
            "material error": rate(rows, "material_error", "yes"),
            "internal contradiction": rate(rows, "contradiction", "yes"),
            "material error, of correct status": rate(correct, "material_error", "yes"),
        }
    )
    return measures


def location_measures(rows):
    """The defect's location on anomalous images that have a location
    reference, and an empty location on normal images."""
    eligible = [
        row
        for row in with_label(rows, "anomalous")
        if row["location_grade"] != "reference unavailable"
    ]
    answered = [row for row in eligible if row["location_grade"] != "missing"]
    return {
        "location answered": share(len(answered), len(eligible)),
        "location correct": rate(eligible, "location_grade", "correct"),
        "location partly correct": rate(eligible, "location_grade", "partial"),
        "location wrong": rate(eligible, "location_grade", "wrong"),
        "location not established": rate(
            eligible, "location_grade", "not established"
        ),
        "location correct, of answered": rate(answered, "location_grade", "correct"),
        "location missing": rate(eligible, "location_grade", "missing"),
        "location appropriately empty, normal images": rate(
            with_label(rows, "normal"), "location_grade", "appropriate n/a"
        ),
    }


def detector_measures(rows):
    """Agreement with the detector and, for each kind of detector evidence,
    how often the reports interpret it and how often fully faithfully."""
    agree = sum(row["inspection_status"] == row["prediction"] for row in rows)
    measures = {"status agrees with the detector": share(agree, len(rows))}
    for column in DETECTOR_GRADES:
        name = column.replace("_", " ")
        offered = [row for row in rows if row[column] != "n/a"]
        attempted = [row for row in offered if row[column] != "unattempted"]
        measures[f"{name}: attempted"] = share(len(attempted), len(offered))
        measures[f"{name}: fully faithful, of attempted"] = rate(
            attempted, column, "fully faithful"
        )
        measures[f"{name}: fully faithful"] = rate(offered, column, "fully faithful")
    return measures


def paired_changes(comparison, before, after, case_ids):
    """Better/same/worse counts of each paired measure over the cases.

    exact_p is the two-sided exact sign test on the cases that changed, which
    is McNemar's exact test for the yes/no measures; it is 1 without a change.
    """
    rows = []
    for measure, value in PAIRED.items():
        changes = {"better": 0, "same": 0, "worse": 0}
        for case_id in case_ids:
            old, new = value(before[case_id]), value(after[case_id])
            changes["better" if new > old else "worse" if new < old else "same"] += 1
        changed = changes["better"] + changes["worse"]
        exact_p = binomtest(changes["better"], changed).pvalue if changed else 1.0
        rows.append(
            {
                "comparison": comparison,
                "measure": measure,
                **changes,
                "exact_p": f"{exact_p:.4f}",
            }
        )
    return rows


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    files = config["files"]
    cases = {case["case_id"]: case for case in read_csv(files["cases"])}
    reports = read_csv(files["reports"])
    if keys(reports) != sorted(itertools.product(cases, config["conditions"])):
        raise SystemExit(f"{files['reports']} needs one report per case and condition")
    sheet = Path(files["grades"])
    if not sheet.exists():
        write_csv(sheet, blank_sheet(reports))
        print(
            f"Wrote an empty grading sheet to {sheet}; grade the reports "
            "and run this script again."
        )
        return
    grades = check_grades(read_csv(sheet), reports, cases)

    rows = {condition: {} for condition in config["conditions"]}
    for report in reports:
        case_id, condition = report["case_id"], report["condition"]
        rows[condition][case_id] = {
            **cases[case_id],
            **report,
            **grades[case_id, condition],
        }

    measures = {}
    for condition, by_case in rows.items():
        graded = list(by_case.values())
        measures[condition] = {
            "valid reports": rate(graded, "valid", "yes"),
            **status_measures(graded, "inspection_status"),
            **grade_measures(graded),
            **location_measures(graded),
        }
        if any(row["detector_prediction"] for row in graded):
            measures[condition].update(detector_measures(graded))
    detector = status_measures(list(cases.values()), "prediction")
    names = dict.fromkeys(name for values in measures.values() for name in values)
    summary = [
        {
            "measure": name,
            **{
                condition: values.get(name, "")
                for condition, values in measures.items()
            },
            "detector": detector.get(name, ""),
        }
        for name in names
    ]

    with_card = [
        case_id for case_id in cases if rows["D1"][case_id]["card_shown"] == "yes"
    ]
    paired = (
        paired_changes("D0 vs I0", rows["I0"], rows["D0"], cases)
        + paired_changes("D1 vs D0", rows["D0"], rows["D1"], cases)
        + paired_changes("D1 vs D0, card shown", rows["D0"], rows["D1"], with_card)
    )

    folder = Path(files["reports"]).parent
    write_csv(folder / "summary.csv", summary)
    write_csv(folder / "paired_changes.csv", paired)
    width = max(map(len, names))
    for row in summary:
        print(
            f"{row['measure']:<{width}}",
            *(f"{value:>14}" for value in list(row.values())[1:]),
        )
    for row in paired:
        print(
            f"{row['comparison']}, {row['measure']}: better "
            f"{row['better']}, same {row['same']}, worse {row['worse']}, "
            f"exact p {row['exact_p']}"
        )
    print(f"Wrote summary.csv and paired_changes.csv to {folder}")


if __name__ == "__main__":
    main()
