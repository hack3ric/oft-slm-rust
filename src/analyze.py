import json
from util import *
import itertools
import tree_sitter_rust as tsrust
from tree_sitter import Language, Parser

RUST_LANGUAGE = Language(tsrust.language())
parser = Parser(RUST_LANGUAGE)


def is_valid_rust_syntax(code: str) -> bool:
    # 2. Parse the generated code
    tree = parser.parse(bytes(code, "utf8"))

    # 3. Check for any syntax errors in the parsed tree
    has_error = False

    def traverse(node):
        nonlocal has_error
        if node.type == "ERROR" or node.is_missing:
            has_error = True
        for child in node.children:
            traverse(child)

    traverse(tree.root_node)
    return not has_error


result_ids = sorted([x for x in range(0, 6920)])

base_correct_count = 0
finetuned_correct_count = 0

with open("results/base.jsonl") as base_file, open(
    "results/finetuned.jsonl"
) as finetuned_file:
    for i in itertools.count():
        base_json_str = base_file.readline()
        finetuned_json_str = finetuned_file.readline()
        if base_json_str == "" or finetuned_json_str == "" or len(result_ids) == 0:
            break
        if result_ids[0] == i:
            result_ids.pop(0)
        else:
            continue
        base = json.loads(base_json_str)
        finetuned = json.loads(finetuned_json_str)
        file_name = base["file_name"]
        prefix = base["prefix"]
        middle_base = base["middle_base"]
        middle_finetuned = finetuned["middle_finetuned"]
        suffix = base["suffix"]
        # print(f"result {i}")
        base_correct = is_valid_rust_syntax(prefix + middle_base + suffix)
        finetuned_correct = is_valid_rust_syntax(prefix + middle_finetuned + suffix)
        # print(f"base correct: {base_correct}")
        # print(f"finetuned correct: {finetuned_correct}")
        if base_correct:
            base_correct_count += 1
        if finetuned_correct:
            finetuned_correct_count += 1

        # print("-" * 25 + " base " + "-" * 25)
        # print_fim(file_name, prefix, middle_base, suffix)
        # print("-" * 25 + " finetuned " + "-" * 25)
        # print_fim(file_name, prefix, middle_finetuned, suffix)

print(f"base correct: {base_correct_count}")
print(f"finetuned correct: {finetuned_correct_count}")
