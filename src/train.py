import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import OFTConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer
from util import *

if __name__ == "__main__":
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Qwen doesn't have a default pad token, so we set it to eos_token
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Setup Orthogonal Finetuning (OFT)
    # Targeting the attention and MLP linear layers for Qwen models
    oft_config = OFTConfig(
        r=8,
        oft_block_size=0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        module_dropout=0.0,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, oft_config)
    model.print_trainable_parameters()

    print("Loading dataset...")
    dataset = (
        load_dataset(dataset_id, split="train")
        .map(format_prompts_from_dataset)
        .shuffle(seed=114514)
    )
    train_dataset = dataset.select(range(40000))

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=50,
        num_train_epochs=2,
        # max_steps=200,          # Increase for actual final training run
        # save_steps=50,
        optim="adamw_torch",
        bf16=True,
        report_to="none",
        dataset_text_field="text",
        save_total_limit=1,
        # max_seq_length=512,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
    )

    print("\nStarting OFT training...")
    # train_result = trainer.train(resume_from_checkpoint=True)
    train_result = trainer.train()

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

    trainer.model.save_pretrained(f"{output_dir}/final_model")
