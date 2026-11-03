# Efficient AI Computing

**New York University (NYU) | Fall 2025**

This repository collects my coursework and final project in **Efficient AI Computing**, focused on making deep learning models more practical under constraints on compute, memory, and inference time.

The work progresses from **neural network pruning** and **quantization** to **efficient language-model decoding**, and culminates in a final project on **knowledge distillation for vision-based cyclist detection**. Across these projects, the central question is how to reduce computational cost while preserving useful model performance.

## Projects at a glance

| Project | Focus | Models / Methods |
| --- | --- | --- |
| [Lab 1: Neural Network Pruning](Lab-1-Pruning/) | Removing less useful weights and structures | ResNet-18, magnitude pruning, L1 regularization, filter pruning |
| [Lab 2: Quantization](Lab-2-Quantization/) | Reducing numerical precision while retaining accuracy | ResNet-18, GPT-2, post-training quantization, quantization-aware training |
| [Lab 3: Speculative Decoding](Lab-3-Speculative-Decoding/) | Accelerating autoregressive language-model inference | Draft/target models, LoRA, token and layer pruning |
| [Final Project: Knowledge Distillation](Knowledge-Distillation/) | Compressing vision models for rare-class cyclist detection | ViT-Large / ViT-Base teachers, ViT-Tiny student, logit/feature/attention KD |

## Lab 1 — Neural Network Pruning

Using **ResNet-18 on CIFAR-10**, this lab examines the accuracy–sparsity trade-off through:

- Analysis of layer weight distributions and sensitivity to pruning.
- Magnitude-based pruning and layer-specific pruning strategies.
- L1 regularization followed by pruning.
- Structured, filter-wise pruning and PyTorch's pruning utilities.

The emphasis is on identifying which parameters can be removed with minimal degradation and understanding why sparsity does not automatically translate into hardware speedups.

**[Explore Lab 1](Lab-1-Pruning/)**

## Lab 2 — Neural Network Quantization

This lab investigates low-precision inference for both **ResNet-18 (CIFAR-10)** and **GPT-2**, including:

- Tensor-wise and filter-wise asymmetric fixed-point quantization.
- Bit-width and clipping-range trade-offs.
- Symmetric quantization and quantization-aware training using a straight-through estimator.
- Tensor-wise, channel-wise, and token-wise quantization for Transformer linear layers.

The experiments explore how quantization granularity and precision affect predictive accuracy, especially at lower bit-widths.

**[Explore Lab 2](Lab-2-Quantization/)**

## Lab 3 — Speculative Decoding for Language Models

This lab studies **speculative decoding**, where a smaller draft model proposes tokens and a larger target model verifies them. It includes:

- Implementation of draft generation, target verification, and token acceptance.
- Supervised fine-tuning of a draft model using **LoRA** and HumanEval-style examples.
- Comparison of latency, speedup, and draft-token acceptance rate.
- Additional experiments on token pruning and Transformer layer pruning.

The experiments highlight an important trade-off: inference can become faster without necessarily preserving the quality of generated text. In the recorded speculative-decoding comparison, outcomes varied with LoRA settings; the best reported configuration reached **1.07× speedup** and **75% acceptance**, while other configurations ran slower than the baseline.

**[Explore Lab 3](Lab-3-Speculative-Decoding/)**

## Final Project — Knowledge Distillation for Vision-Based Cyclist Detection

**[Explore the full project, report, and notebook](Knowledge-Distillation/)**

The final project investigates whether a compact Vision Transformer can retain much of a larger model's bicycle-recognition performance in a resource-constrained setting. **CIFAR-100** is reformulated as an imbalanced binary task: bicycle versus all other classes, with approximately **1% positive examples**.

I compared **logit**, **intermediate-feature**, and **attention-map distillation** from ViT-Large and ViT-Base teachers to a **ViT-Tiny student**, along with a teacher-assistant pipeline using ViT-Small.

### Selected results

| Model / approach | Bicycle-class F1 |
| --- | ---: |
| ViT-Tiny without distillation | 0.856 |
| ViT-Tiny + logit KD from ViT-Large | 0.910 |
| **ViT-Tiny + logit KD from ViT-Base** | **0.922** |
| ViT-Tiny + feature KD from ViT-Large | 0.905 |
| ViT-Tiny + attention KD from ViT-Large | 0.901 |

The strongest student used **5.52 million parameters** and achieved **0.922 F1**, compared with **0.856** for the undistilled baseline. Under these experimental conditions, simple logit distillation outperformed the more elaborate alternatives.

**Scope:** This is a controlled image-classification proxy for cyclist detection, **not a validated automotive blind-spot detection system**. The project README and report discuss data-pipeline limitations, run-to-run variance, and the need for evaluation on real driving scenes.

## Tools and technologies

Python · PyTorch · Torchvision · Hugging Face Transformers · `timm` · Jupyter / Google Colab · LoRA / PEFT

The labs and final project use different model architectures and dependencies; see the notebooks and each project's documentation for setup details. **GPU access is recommended or required** for the training and evaluation workflows.

## Repository organization

```text
Efficient-AI-Computing/
├── README.md
├── Lab-1-Pruning/
├── Lab-2-Quantization/
├── Lab-3-Speculative-Decoding/
└── Knowledge-Distillation/
    ├── README.md
    ├── notebooks/
    │   └── vit_kd_bicycle_detection.ipynb
    └── report/
        ├── report.pdf
        └── report.md
```

*The lab folders shown above are the intended organization; their links will work after the notebooks have been uploaded to those paths. The final-project folder may contain additional files as they are added.*

## Acknowledgment

The three lab notebooks are based on course-provided assignments and starter material, with my implementations, experiments, and written analyses. The knowledge-distillation project is my course final project.
