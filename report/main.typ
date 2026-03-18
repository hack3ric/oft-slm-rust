= AIST5030 Mini-Project Report

== Introduction

The advent of AI-driven code generation tools like GitHub Copilot has established Large Language Models (LLMs) as highly effective assistants for software development. These tools have shifted code generation from simple line-by-line prediction to complex context-aware completion. A core mechanism used by LLM code generation tool is the Fill-in-the-Middle (FIM) task, where a model must intelligently generate missing code bridging the gap between an existing preceding context (prefix) and succeeding context (suffix).

In the mini-project, we finetuned a small LLM to complete FIM tasks specifically for Rust, a modern, memory-safe programming language, using the parameter-efficient Orthogonal Finetuning (OFT) method.

== Methodology

=== Model and Dataset

- Model: Qwen2.5-1.5B-Instruct
- Dataset: Etherll/CodeFIM-Rust-Mellum

=== OFT Configuration

== Experiment

=== Setup

1 RTX 6000 Ada for training. The training time is \~4.5 hours. 40k items are used for training, \~7k for evaluation

#figure(caption: "Example prompt for Fill-in-the-Middle tasks")[```
<filename>main.rs
<fim_suffix>}
<fim_prefix>fn fibonacci(n: i32) -> i32 {
<fim_middle>
```]

TODO: use tree-sitter to verify syntactic correctness

== Discussion and Limitations

The dataset contains archaic, pre-alpha Rust code dated back to 2012, while the first stable version, 1.0, was released in 2015. These code snippets contain syntaxes that does not exist in current stable versions. This holds back the finetuned model's capability to generate correct Rust code.

== Conclusion

// Table of Contents (TOC)

// 1. Introduction
// 2. Methodology
//     2.1 Model and Dataset Selection
//     2.2 Orthogonal Finetuning (OFT) Configuration
// 3. Experimental Results
//     3.1 Training Performance (Loss Curve)
//     3.2 Qualitative Evaluation (Before & After)
// 4. Discussion & Limitations
//     4.1 Format Adherence vs. Logical Generalization
//     4.2 Data Quality & Archaic Syntax
// 5. Conclusion
