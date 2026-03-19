model_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_path = "./qwen-oft-rust/final_model"  # Path where your trainer saved the model
dataset_id = "Etherll/CodeFIM-Rust-Mellum"
output_dir = "./qwen-oft-rust"


def format_prompts_from_dataset(example):
    return {
        "text": format_prompts_str(
            example.get("file_name", ""),
            example.get("prefix", ""),
            example.get("suffix", ""),
            example.get("middle", ""),
        )
    }


def format_prompts_from_dataset_input(example):
    return {
        "text": format_prompts_str(
            example.get("file_name", ""),
            example.get("prefix", ""),
            example.get("suffix", ""),
        )
    }


def format_prompts_str(file_name, prefix, suffix, middle=""):
    return f"<filename>{file_name}<fim_suffix>{suffix}<fim_prefix>{prefix}<fim_middle>{middle}"


def print_fim(file_name, prefix, middle, suffix):
    # print("```")
    print(f"// {file_name}\n")
    print(f"{prefix}\033[33m{middle}\033[0m\n{suffix}")
    # print("```")
