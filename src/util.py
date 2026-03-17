def format_prompts_from_dataset(example):
    return {"text": format_prompts_str(example.get('file_name', ''), example.get('prefix', ''), example.get('suffix', ''), example.get('middle', ''))}

def format_prompts_str(file_name, prefix, suffix, middle=""):
    return f"<filename>{file_name}<fim_suffix>{suffix}<fim_prefix>{prefix}<fim_middle>{middle}"
