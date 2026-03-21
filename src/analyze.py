import json

import Levenshtein
import evaluate
from util import *
import itertools
import tree_sitter_rust as tsrust
from tree_sitter import Language, Parser

# from datasets import load_dataset

RUST_LANGUAGE = Language(tsrust.language())
parser = Parser(RUST_LANGUAGE)


def is_valid_rust_syntax(code: str) -> bool:
    tree = parser.parse(bytes(code, "utf8"))

    has_error = False

    def traverse(node):
        nonlocal has_error
        if node.type == "ERROR" or node.is_missing:
            has_error = True
        for child in node.children:
            traverse(child)

    traverse(tree.root_node)
    return not has_error


# result_ids = sorted([x for x in range(0, 100)])

# dataset = load_dataset(dataset_id, split="train").shuffle(seed=114514)
# test_dataset = [x for x in dataset][50000:]
# results = [None] * len(test_dataset)

results = []

bleu_metric = evaluate.load("bleu")
count = 0
with open("results/base.jsonl") as base_file, open(
    "results/finetuned.jsonl"
) as finetuned_file:
    for i in itertools.count():
        base_json_str = base_file.readline()
        finetuned_json_str = finetuned_file.readline()
        if base_json_str == "" or finetuned_json_str == "":
            break
        count += 1
        # if len(result_ids) == 0:
        #     break
        # if result_ids[0] == i:
        #     result_ids.pop(0)
        # else:
        #     continue
        base = json.loads(base_json_str)
        finetuned = json.loads(finetuned_json_str)
        file_name = base["file_name"]
        prefix = base["prefix"]
        middle_ground = base["middle_ground"]
        middle_base = base["middle_base"]
        middle_finetuned = finetuned["middle_finetuned"]
        suffix = base["suffix"]
        base["middle_ground"] = middle_ground

        ground_correct = is_valid_rust_syntax(prefix + middle_ground + suffix)
        base_correct = is_valid_rust_syntax(prefix + middle_base + suffix)
        finetuned_correct = is_valid_rust_syntax(prefix + middle_finetuned + suffix)

        middle_ground = middle_ground.strip()
        middle_base = middle_base.strip()
        middle_finetuned = middle_finetuned.strip()

        # if ground_correct:
        #     ground_correct_items.append(i)
        # if base_correct:
        #     base_correct_items.append(i)
        # if finetuned_correct:
        #     finetuned_correct_items.append(i)

        base_edit_ratio = Levenshtein.ratio(middle_ground, middle_base)
        finetuned_edit_ratio = Levenshtein.ratio(middle_ground, middle_finetuned)

        results.append(
            {
                "middle_ground": middle_ground,
                "middle_base": middle_base,
                "middle_finetuned": middle_finetuned,
                "ground_correct": ground_correct,
                "base_correct": base_correct,
                "finetuned_correct": finetuned_correct,
                "base_edit_ratio": base_edit_ratio,
                "finetuned_edit_ratio": finetuned_edit_ratio,
            }
        )

        # print("-" * 25 + " base " + "-" * 25)
        # print_fim(file_name, prefix, middle_base, suffix)
        # print("-" * 25 + " finetuned " + "-" * 25)
        # print_fim(file_name, prefix, middle_finetuned, suffix)

print(f"ground correct: {len([r for r in results if r['ground_correct']])}")
print(f"base correct: {len([r for r in results if r['base_correct']])}")
print(f"finetuned correct: {len([r for r in results if r['finetuned_correct']])}")

intersection_base = set(
    [i for i, r in enumerate(results) if r["ground_correct"] and r["base_correct"]]
)
intersection_finetuned = set(
    [i for i, r in enumerate(results) if r["ground_correct"] and r["finetuned_correct"]]
)
print(f"intersection ground and base: {len(intersection_base)}")
print(f"intersection ground and finetuned: {len(intersection_finetuned)}")

bleu_base = bleu_metric.compute(
    predictions=[r["middle_base"] for r in results],
    references=[[r["middle_ground"]] for r in results],
)["bleu"]
bleu_finetuned = bleu_metric.compute(
    predictions=[r["middle_finetuned"] for r in results],
    references=[[r["middle_ground"]] for r in results],
)["bleu"]
print(f"BLEU score for base model: {bleu_base:.4f}")
print(f"BLEU score for finetuned model: {bleu_finetuned:.4f}")

print(f"Average edit ratio for base model: {sum(r['base_edit_ratio'] for r in results) / len(results):.4f}")
print(f"Average edit ratio for finetuned model: {sum(r['finetuned_edit_ratio'] for r in results) / len(results):.4f}")
