"""Generate the inspection reports with Qwen2.5-VL-3B-Instruct.

One report per selected image and condition (Bai et al., 2025):

    I0  the image only;
    D0  plus Dinomaly's prediction, score and cutoff, and its anomaly map
        over the image when the prediction is "anomalous";
    D1  plus the category's own retrieved card, when there is one.

The language layers run in 4-bit NF4 (Dettmers et al., 2023); decoding is
greedy. Texts and settings are in configs/reporting.json and
configs/prompt.txt. The answer must be one JSON object that passes
configs/report_schema.json once an optional ```json fence is stripped.
Ground-truth labels never reach the model.

Needs a CUDA GPU with bfloat16. Writes outputs/reports/reports.csv and
overlays/<case>.png, and stops if reports.csv already exists.
Run from the repository root: python -m scripts.07_generate_reports
"""

import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from jsonschema import Draft202012Validator
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from src.datasets import DATA_ROOTS, list_samples, read_image
from src.predictions import load_category
from src.visualize import draw_overlay

CONFIG = Path("configs/reporting.json")
FENCE = re.compile(r"\A```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```[ \t]*\Z")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_model(settings):
    """Qwen2.5-VL in 4-bit NF4 on the first GPU, and its processor."""
    quantization = dict(settings["quantization"])
    for key in ("bnb_4bit_compute_dtype", "bnb_4bit_quant_storage"):
        quantization[key] = getattr(torch, quantization[key])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        settings["id"],
        revision=settings["revision"],
        quantization_config=BitsAndBytesConfig(**quantization),
        dtype=getattr(torch, settings["dtype"]),
        device_map=settings["device_map"],
        attn_implementation=settings["attn_implementation"],
    )
    return model.eval()


def build_messages(texts, prompt, case, condition, image, overlay, card_block):
    """The chat messages of one report; images are PIL images."""
    content = [
        {"type": "text", "text": texts["original_image"]},
        {"type": "image", "image": image},
    ]
    if condition != "I0":
        content.append(
            {
                "type": "text",
                "text": texts["detector"].format(
                    prediction=case["prediction"],
                    score=float(case["score"]),
                    cutoff=float(case["cutoff"]),
                ),
            }
        )
        if overlay is not None:
            content += [
                {"type": "text", "text": texts["overlay_title"]},
                {"type": "image", "image": overlay},
                {"type": "text", "text": texts["overlay_note"]},
            ]
    if condition == "D1" and card_block:
        content.append({"type": "text", "text": texts["section_break"] + card_block})
    content.append({"type": "text", "text": texts["section_break"] + prompt})
    return [
        {"role": "system", "content": texts["system"]},
        {"role": "user", "content": content},
    ]


def case_inputs(case, settings, cards, retrieved):
    """The image, overlay (or None), card text and prompt of a case."""
    dataset, category = case["dataset"], case["category"]
    sample = next(
        s
        for s in list_samples(dataset, DATA_ROOTS[dataset], category, "test")
        if s.image_id == case["image_id"]
    )
    image = read_image(sample.image_path)

    overlay = None
    if case["prediction"] == "anomalous":
        saved = load_category(settings["detector"]["method"], dataset, category)
        i = saved["image_ids"].tolist().index(case["image_id"])
        style = settings["overlay"]
        overlay = Image.fromarray(
            draw_overlay(
                np.asarray(image),
                saved["anomaly_maps"][i],
                saved["map_boxes"][i],
                style["color_rgb"],
                style["max_opacity"],
                style["outline_gray"],
            )
        )

    card_block = ""
    if retrieved[(dataset, category)]["used_in_d1"] == "yes":
        card = cards[(dataset, category)]
        card_block = settings["messages"]["card"].format(
            card_text=card["card_text"], sources=card["sources"]
        )

    template = Path(settings["files"]["prompt"]).read_text(encoding="utf-8")
    prompt = template.replace("{category}", category.replace("_", " "))
    return image, overlay, card_block, prompt


def model_inputs(processor, messages, settings):
    """Token IDs and image tensors, as the Qwen processor makes them."""
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=settings["processor"]["add_generation_prompt"],
        processor_kwargs={
            "images_kwargs": {"size": settings["processor"]["image_pixels"]},
            "text_kwargs": settings["processor"]["text_kwargs"],
        },
    )


def generate(model, processor, messages, settings):
    """Qwen's answer as text, without the end-of-turn token."""
    inputs = model_inputs(processor, messages, settings)
    with torch.inference_mode():
        output = model.generate(
            **{name: value.to(model.device) for name, value in inputs.items()},
            **settings["generation"],
        )
    tokens = output[0, inputs["input_ids"].shape[1] :].tolist()
    end_tokens = model.generation_config.eos_token_id
    if isinstance(end_tokens, int):
        end_tokens = [end_tokens]
    if tokens and tokens[-1] in end_tokens:
        tokens = tokens[:-1]
    return processor.tokenizer.decode(
        tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


def _unique_keys(pairs):
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate key")
    return dict(pairs)


def _finite(text):
    number = float(text)
    if not math.isfinite(number):
        raise ValueError(f"number out of range: {text}")
    return number


def _no_constant(name):
    raise ValueError(f"not a JSON number: {name}")


def parse_report(text, schema):
    """The parsed report (or None) and what is wrong with it ("" if valid).

    Strict JSON: no repairs, duplicate keys and NaN/Infinity are rejected.
    """
    payload = text.strip()
    if not payload:
        return None, "empty answer"
    match = FENCE.fullmatch(payload)
    if match:
        payload = match.group(1)
    try:
        report = json.loads(
            payload,
            object_pairs_hook=_unique_keys,
            parse_float=_finite,
            parse_constant=_no_constant,
        )
    except ValueError as error:
        return None, f"not valid JSON: {error}"
    errors = [
        error.message for error in Draft202012Validator(schema).iter_errors(report)
    ]
    return report, "; ".join(errors)


def main():
    settings = json.loads(CONFIG.read_text(encoding="utf-8"))
    files = settings["files"]
    schema = json.loads(Path(files["schema"]).read_text(encoding="utf-8"))
    fields = schema["required"]
    cards = {
        (card["dataset"], card["category"]): card
        for card in read_csv(files["knowledge"])
    }
    retrieved = {
        (row["dataset"], row["category"]): row
        for row in read_csv(files["retrieved_cards"])
    }
    output = Path(files["reports"])
    if output.exists():
        raise SystemExit(f"{output} exists; move or delete it before a new run")
    (output.parent / "overlays").mkdir(parents=True, exist_ok=True)

    model = load_model(settings["model"])
    processor = AutoProcessor.from_pretrained(
        settings["model"]["id"],
        revision=settings["model"]["revision"],
        backend=settings["processor"]["backend"],
    )

    cases = read_csv(files["cases"])
    columns = [
        "case_id",
        "condition",
        "dataset",
        "category",
        "image_id",
        "detector_prediction",
        "overlay_shown",
        "card_shown",
        "valid",
        "problem",
        *fields,
        "raw_output",
    ]
    with open(output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for case in cases:
            dataset, category = case["dataset"], case["category"]
            image, overlay, card_block, prompt = case_inputs(
                case, settings, cards, retrieved
            )
            if overlay is not None:
                overlay.save(output.parent / "overlays" / f"{case['case_id']}.png")

            for condition in settings["conditions"]:
                messages = build_messages(
                    settings["messages"],
                    prompt,
                    case,
                    condition,
                    image,
                    overlay,
                    card_block,
                )
                answer = generate(model, processor, messages, settings)
                report, problem = parse_report(answer, schema)
                report = report if isinstance(report, dict) else {}
                writer.writerow(
                    {
                        "case_id": case["case_id"],
                        "condition": condition,
                        "dataset": dataset,
                        "category": category,
                        "image_id": case["image_id"],
                        "detector_prediction": (
                            "" if condition == "I0" else case["prediction"]
                        ),
                        "overlay_shown": (
                            "yes" if condition != "I0" and overlay is not None else "no"
                        ),
                        "card_shown": (
                            "yes" if condition == "D1" and card_block else "no"
                        ),
                        "valid": "no" if problem else "yes",
                        "problem": problem,
                        **{field: report.get(field) for field in fields},
                        "raw_output": answer,
                    }
                )
                file.flush()  # keep finished reports if a later one fails
                print(
                    f"{case['case_id']} {condition}: "
                    f"{'valid' if not problem else 'invalid: ' + problem}"
                )
    print(f"Wrote {len(cases) * len(settings['conditions'])} reports to {output}")


if __name__ == "__main__":
    main()
