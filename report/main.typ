= AIST5030 Mini-Project Report



== Introduction

The advent of AI-driven code generation tools like GitHub Copilot has established Large Language Models (LLMs) as highly effective assistants for software development. These tools have shifted code generation from simple line-by-line prediction to complex context-aware completion. A core mechanism used by LLM code generation tool is the Fill-in-the-Middle (FIM) task, where a model must intelligently generate missing code bridging the gap between an existing preceding context (prefix) and succeeding context (suffix).

In the mini-project, we finetuned a small LLM to complete FIM tasks specifically for Rust, a modern, memory-safe programming language, using the parameter-efficient Orthogonal Finetuning (OFT) method.

== Methodology

=== Fill-in-the-Middle Tasks

@code-editor

A basic prompt structure used for FIM is shown in @fim-prompt, which includes the current open file's name, code before and after the cursor

#block(height: 10em)[
  #set align(horizon)
  #grid(columns: (1fr,) * 2, column-gutter: 1em)[
    #figure(caption: "Example code editor state in an IDE")[
      #block(height: 10em)[
        #box(stroke: 1pt + gray)[#image("images/vscode.png")]]
    ] <code-editor>
  ][
    #figure(caption: "Derived FIM prompt")[
      #block(height: 10em)[#align(left)[
        #set text(font: "JetBrains Mono", size: 0.8em)
        #text(fill: blue)[\<filename>]main.rs\
        #text(fill: blue)[\<fim_suffix>]}\
        #text(fill: blue)[\<fim_prefix>]fn fibonacci(n: i32) -> i32 {\
        #text(fill: blue)[\<fim_middle>]
      ]]
    ] <fim-prompt>
  ]
]


=== Model and Dataset

- Model: Qwen2.5-1.5B-Instruct
- Dataset: Etherll/CodeFIM-Rust-Mellum
  - Contains \~57,000 data items

=== OFT Configuration

Rank r=8, target modules: attention and MLP layers, Learning Rate: 2e-4, 2 epochs totaling 2,500 steps

== Experiment

=== Setup

1 RTX 6000 Ada for training. The training time is \~4.5 hours. 40,000 items are used for training, \~7,000 for evaluation

=== Training

#figure(caption: "Loss curve")[
  #image("images/loss_curve.png", width: 50%)
]

=== Qualitative Evaluation

#figure(caption: "Example of errorneous base model output (left) vs correct finetuned model output (right)")[
  #block(height: 14em)[
    #grid(columns: 2, column-gutter: 1em)[
      #box(stroke: 1pt + gray)[#image("images/base_error_fim.png")]
    ][
      #box(stroke: 1pt + gray)[#image("images/finetuned_good_fim.png")]
    ]
  ]
]



TODO: use tree-sitter to verify syntactic correctness

#figure(caption: "Qualitative evaluation results (higher is better)")[
  #set text(size: 0.8em)
  #table(
    columns: 4,
    table.header([], [Syntactic Correctness], [Average Levenshtein Edit Ratio], [BLEU Score]),
    [Ground Truth], [0.7773], [-], [-],
    [Base Model], [0.0804], [0.1817], [0.0590],
    [Finetuned Model], [0.6046], [0.5729], [0.2891],
    [Improvement], [7.5x], [3.15x], [4.9x],
  )
]

Since long prefixes and suffixes are stitched to model output, the original CodeBLEU scores are close to 1. Therefore, they are inverted using the formula $N' = 100(1 - N)$.

// #figure(caption: "CodeBLEU scores")[
//   #set text(size: 0.8em)
//   #table(
//     columns: 6,
//     table.header(
//       [], [CodeBLEU], [N-Gram Match], [Weighted N-Gram Match], [Syntax Match], [Dataflow Match],
//       [Base Model], [0.9088], [0.8792], [0.9816], [0.9506], [0.8237],
//       [Finetuned Model], [0.9701], [0.9774], [0.9828], [0.9783], [0.9419],
//       [Improvement], [1.06x], [], [], [], [],
//     ),
//   )
// ]

#figure(caption: "CodeBLEU scores (inverted, lower is better)")[
  #set text(size: 0.8em)
  #table(
    columns: 6,
    table.header(
      [], [CodeBLEU], [N-Gram Match], [Weighted N-Gram Match], [Syntax Match], [Dataflow Match],
      [Base Model], [9.12], [10.28], [1.84], [4.94], [17.63],
      [Finetuned Model], [2.99], [2.26], [1.72], [2.17], [5.81],
      [Error Reduction], [67.21%], [78.01%], [6.52%], [56.07%], [67.04%],
    ),
  )
]

== Discussion and Limitations

The dataset contains archaic, pre-alpha Rust code dated back to 2012, while the first stable version, 1.0, was released in 2015. These code snippets contain syntaxes that does not exist in current stable versions. This holds back the finetuned model's capability to generate correct Rust code.

Prompt injection: control string is not escaped

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
