"""Retrieve one knowledge card per category for the D1 reports.

E5-small-v2 (Wang et al., 2022) embeds the 27 cards of
knowledge/defect_cards.csv as "passage: " texts and one query per category as
a "query: " text, as in its model card (mean pooling, unit length). An exact
FAISS inner-product search finds each query's best card; D1 uses it only if
it is the category's own card. Settings: configs/reporting.json.

Writes outputs/retrieval/retrieved_cards.csv.
Run from the repository root: python -m scripts.06_retrieve_knowledge
"""

import csv
import json
from pathlib import Path

import faiss
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

CONFIG = Path("configs/reporting.json")


# average_pool() is the function of the same name in the usage example of
# the intfloat/e5-small-v2 model card, Copyright (c) Microsoft Corporation,
# MIT License.
def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


@torch.no_grad()
def embed(texts, tokenizer, model):
    """Unit-length E5 embeddings, as in the model card example."""
    batch = tokenizer(
        texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )
    outputs = model(**batch)
    embeddings = average_pool(outputs.last_hidden_state, batch["attention_mask"])
    return F.normalize(embeddings, p=2, dim=1).numpy()


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    settings = config["retrieval"]
    with open(config["files"]["knowledge"], newline="", encoding="utf-8") as file:
        cards = list(csv.DictReader(file))

    tokenizer = AutoTokenizer.from_pretrained(
        settings["model"], revision=settings["revision"]
    )
    model = AutoModel.from_pretrained(
        settings["model"], revision=settings["revision"]
    ).eval()
    passages = embed(
        [settings["passage_prefix"] + card["card_text"] for card in cards],
        tokenizer,
        model,
    )
    queries = [
        settings["query"].format(
            dataset=card["dataset"], category=card["category"].replace("_", " ")
        )
        for card in cards
    ]
    found = embed(
        [settings["query_prefix"] + query for query in queries], tokenizer, model
    )

    index = faiss.IndexFlatIP(passages.shape[1])
    index.add(passages)
    similarities, best = index.search(found, 1)

    rows = []
    for card, query, similarity, row in zip(
        cards, queries, similarities[:, 0], best[:, 0]
    ):
        retrieved = cards[row]
        used = (retrieved["dataset"], retrieved["category"]) == (
            card["dataset"],
            card["category"],
        )
        rows.append(
            {
                "dataset": card["dataset"],
                "category": card["category"],
                "query": query,
                "retrieved_card": f"{retrieved['dataset']}/{retrieved['category']}",
                "similarity": f"{similarity:.4f}",
                "used_in_d1": "yes" if used else "no",
            }
        )
        print(
            f"{card['dataset']}/{card['category']}: "
            f"{rows[-1]['retrieved_card']} ({similarity:.4f})"
            f"{'' if used else ', not used'}"
        )

    output = Path(config["files"]["retrieved_cards"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    own_cards = sum(row["used_in_d1"] == "yes" for row in rows)
    print(f"{own_cards} of {len(rows)} categories get a card -> {output}")


if __name__ == "__main__":
    main()
