import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from util import *

# --- 1. Settings ---
base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_path = "./qwen-oft-rust/final_model"  # Path where your trainer saved the model

# Test prompts (replace with your own Rust coding questions)
test_prompts = [
    format_prompts_str("factorial.rs", "fn calculate_factorial(n: u64) -> u64 {", "}"),
    format_prompts_str("user.rs", "struct User {\n    username: String,\n    age: u8,\n}\nimpl User {\n", "}"),
]

# --- 2. Load Tokenizer and Base Model ---
print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Helper function to generate code
def generate_code(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        # Adjust max_new_tokens if you want longer/shorter code snippets
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- 3. Generate Responses from BASE Model ---
print("\n" + "="*50)
print("             BASE MODEL RESPONSES")
print("="*50)

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
print("\n" + "="*50)
print("           FINETUNED MODEL RESPONSES")
print("="*50)

finetuned_responses = []
for i, prompt in enumerate(test_prompts):
    result = generate_code(finetuned_model, tokenizer, prompt)
    finetuned_responses.append(result)
    print(f"\n--- Prompt {i+1} ---")
    print(result)

