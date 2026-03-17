import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import OFTConfig, get_peft_model, TaskType
from trl import SFTTrainer

# 1. Configuration
model_id = "Qwen/Qwen2.5-1.5B-Instruct" # Fits the 1-1.5B constraint
dataset_id = "Etherll/CodeFIM-Rust-Mellum"
output_dir = "./qwen-oft-rust"

# 2. Load Tokenizer & Model
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Qwen doesn't have a default pad token, so we set it to eos_token
tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 3. Setup Orthogonal Finetuning (OFT)
# Targeting the attention and MLP linear layers for Qwen models
oft_config = OFTConfig(
    r=8, 
    oft_block_size=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    module_dropout=0.0,
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, oft_config)
model.print_trainable_parameters()

# 4. Load and Format Dataset
print("Loading dataset...")
dataset = load_dataset(dataset_id, split="train")

# Since FIM (Fill-in-the-Middle) datasets often have 'prefix', 'middle', 'suffix' columns,
# this function creates standard causal language modeling text sequences.
# Adjust the column names if your dataset structure differs.
def format_prompts(example):
    # Standard format combining context and the missing Rust code
    text = f"// Context:\n{example.get('prefix', '')}\n// Complete the code:\n{example.get('middle', '')}\n{example.get('suffix', '')}"
    return {"text": text}

dataset = dataset.map(format_prompts)

# Take a small subset for quick prototyping (remove this for full training)
train_dataset = dataset.select(range(5000))

# 5. Training Arguments (Save logs for your report's loss curves)
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=200,          # Increase for actual final training run
    save_steps=50,
    optim="adamw_torch",
    bf16=True,              # Use bf16 if your GPU supports it (Ampere or newer)
    report_to="none"        # Or set to "wandb" to easily export loss curves
)

# 6. Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=512,      # Truncate context for memory efficiency
)

# 7. Pre-Finetuning Qualitative Test (For your report)
print("\n--- BEFORE FINETUNING ---")
test_prompt = "// Context:\nfn main() {\n// Complete the code:\n"
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    pre_outputs = model.generate(**inputs, max_new_tokens=30)
print(tokenizer.decode(pre_outputs[0], skip_special_tokens=True))

# 8. Train the model
print("\nStarting OFT training...")
train_result = trainer.train()

# 9. Extract and Plot Training Loss (Deliverable requirement)
print("\nPlotting training loss curve...")
loss_history = [log["loss"] for log in trainer.state.log_history if "loss" in log]
steps = [log["step"] for log in trainer.state.log_history if "loss" in log]

plt.figure(figsize=(8, 5))
plt.plot(steps, loss_history, label="Training Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("OFT Finetuning Loss Curve")
plt.legend()
plt.savefig(f"{output_dir}/loss_curve.png")
print(f"Loss curve saved to {output_dir}/loss_curve.png")

# 10. Save the Finetuned Weights
trainer.model.save_pretrained(f"{output_dir}/final_model")

# 11. Post-Finetuning Qualitative Test (For your report)
print("\n--- AFTER FINETUNING ---")
with torch.no_grad():
    post_outputs = model.generate(**inputs, max_new_tokens=30)
print(tokenizer.decode(post_outputs[0], skip_special_tokens=True))
