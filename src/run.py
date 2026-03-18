import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from util import *
from datasets import load_dataset

# --- 1. Settings ---
base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_path = "./qwen-oft-rust/final_model"  # Path where your trainer saved the model
dataset_id = "Etherll/CodeFIM-Rust-Mellum"

print("Loading dataset...")
dataset = (
    load_dataset(dataset_id, split="train")
    .map(format_prompts_from_dataset_input)
    .shuffle(seed=114514)
)

# Test prompts (replace with your own Rust coding questions)
test_prompts = [
    dataset[50000]["text"],
    dataset[50001]["text"],
]

# --- 2. Load Tokenizer and Base Model ---
print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, torch_dtype=torch.bfloat16, device_map="auto"
)


# Helper function to generate code
def generate_code(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        # Adjust max_new_tokens if you want longer/shorter code snippets
        outputs = model.generate(
            **inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# --- 3. Generate Responses from BASE Model ---
print("\n" + "=" * 50)
print("             BASE MODEL RESPONSES")
print("=" * 50)

base_responses = []
for i, prompt in enumerate(test_prompts):
    result = generate_code(base_model, tokenizer, prompt)
    base_responses.append(result)
    print(f"\n--- Prompt {i+1} ---")
    print(result)

# --- 4. Load the Finetuned Weights (Adapters) ---
print("\nLoading PEFT adapters...")
# This attaches your tiny OFT weights onto the base model
finetuned_model = PeftModel.from_pretrained(base_model, adapter_path)

# --- 5. Generate Responses from FINETUNED Model ---
print("\n" + "=" * 50)
print("           FINETUNED MODEL RESPONSES")
print("=" * 50)

finetuned_responses = []
for i, prompt in enumerate(test_prompts):
    result = generate_code(finetuned_model, tokenizer, prompt)
    finetuned_responses.append(result)
    print(f"\n--- Prompt {i+1} ---")
    print(result)
