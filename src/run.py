import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from util import *
from datasets import load_dataset


# Helper function to generate code
def fill_in_middle(model, tokenizer, file_name, prefix, suffix):
    prompt = format_prompts_str(file_name, prefix, suffix)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs_len = inputs.input_ids.shape[-1]
    with torch.no_grad():
        # Adjust max_new_tokens if you want longer/shorter code snippets
        outputs = model.generate(
            **inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0, inputs_len:], skip_special_tokens=True)


if __name__ == "__main__":
    print("Loading dataset...")
    dataset = load_dataset(dataset_id, split="train").shuffle(seed=114514)
    test_dataset = [x for x in dataset][50000:]
    results = [None] * len(test_dataset)

    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    os.makedirs("results", exist_ok=True)

    for i, data in enumerate(test_dataset):
        file_name = data.get("file_name", "")
        prefix = data.get("prefix", "")
        suffix = data.get("suffix", "")
        middle = fill_in_middle(base_model, tokenizer, file_name, prefix, suffix)
        results[i] = {
            "file_name": file_name,
            "prefix": prefix,
            "suffix": suffix,
            "middle_ground": data.get("middle", ""),
            "middle_base": middle,
        }
        with open("results/base.jsonl", "a") as file:
            json.dump(results[i], file)
            file.write("\n")
        print(f"base: {i+1}/{len(test_dataset)} done")

    print("\nLoading PEFT adapters...")
    # This attaches your tiny OFT weights onto the base model
    finetuned_model = PeftModel.from_pretrained(base_model, adapter_path)

    for i, data in enumerate(test_dataset):
        file_name = results[i]["file_name"]
        prefix = results[i]["prefix"]
        suffix = results[i]["suffix"]
        middle = fill_in_middle(finetuned_model, tokenizer, file_name, prefix, suffix)
        results[i]["middle_finetuned"] = middle
        with open("results/finetuned.jsonl", "a") as file:
            json.dump({"middle_finetuned": middle}, file)
            file.write("\n")
        print(f"finetuned: {i+1}/{len(test_dataset)} done")
