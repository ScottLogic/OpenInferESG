# Methodology: Calculating Per-Query Energy for LLM Inference

> To view this as a markdown preview in VSCode press `Ctrl+Shift+V`

## Table of Contents

- [Introduction](#introduction)
- [1. Hardware Specifications](#hardware-specifications)
- [2. Core Concept](#core-concept)
- [3. Core Equations and Variables](#core-equations-and-variables)
- [4. Practical Example: GPT-4o](#practical-example-gpt-4o)
- [5. References](#references)

## Introduction

This methodology is based on the research paper by Jegham et al. (2025) _"How Hungry is AI? A Systematic Methodology for Empirical Energy and Water Consumption Analysis of Large Language Models"_ [1] which presents a framework for estimating the energy consumption of Large Language Models (LLMs) during inference in data center environments. The approach provides a systematic way to calculate per-query energy usage, taking into account various hardware and deployment factors. [Section 1](#hardware-specification) provides current hardware specifications for major cloud LLM deployments, which can be used as reference values in the calculations.

## 1. Hardware Specifications

This section provides reference values for hardware configurations used in major cloud LLM deployments. These specifications can be used as input parameters for the energy calculations detailed in the following sections.

### 1.1 Deployment Specifications

**Table 1 – Deployment and Infrastructure Specifications (OpenAI Models)**

| Models                                                                                                                   | Launch Dates        | Host            | Hardware          | Critical Power (kW) | PUE      |
| ------------------------------------------------------------------------------------------------------------------------ | ------------------- | --------------- | ----------------- | ------------------- | -------- |
| GPT-4.1, GPT-4.1 mini, GPT-4.1 nano, o4-mini (high), GPT-4.5, o3, o3-mini (high), o3-mini, o1, o1-mini, GPT-4o (Mar ’25) | Apr 2025 – May 2024 | Microsoft Azure | DGX H200/H100 [2] | 10.20 [3]           | 1.12 [4] |
| GPT-4o mini, GPT-4 Turbo, GPT-4                                                                                          | Jul 2024 – Mar 2023 | Microsoft Azure | DGX A100\*          | 6.50 [5]            | 1.12     |

*DGX A100 was estimated for GPT-4o mini, GPT-4 Turbo, and GPT-4. Justification and estimation details are provided in Section 4.3.1. of _"How Hungry is AI?"_

## 2. Core Concept

The methodology estimates energy consumption per query through three key steps:

1. Calculate the inference time for the request (in hours)
2. Multiply by the **effective system power** (combining GPU and non-GPU components)
3. Multiply by the data center's Power Usage Effectiveness (PUE)

## 3. Core Equations and Variables

### 2.1 Energy Consumption Formula

The core equation for calculating per-query energy consumption is:

```math
E_{\text{query}}\ (\text{kWh}) =
\left(
\frac{\frac{\text{OutputLength}}{\text{TPS}} + \text{Latency}}{3600}
\right)
\cdot
\Big[
P_{\text{GPU}}\cdot U_{\text{GPU\_total}}
+ P_{\text{non-GPU}}\cdot U_{\text{non-GPU\_total}}
\Big]
\cdot \text{PUE}
```

### 2.2 Utilization Formulas

The utilization components are calculated as:

```math
U_{\text{GPU\_total}} = \frac{G \cdot D_{\text{GPU}}}{N \cdot B}
\qquad,\qquad
U_{\text{non-GPU\_total}} = \frac{G \cdot D_{\text{non-GPU}}}{N \cdot B}
```

### 2.3 Variable Definitions and Expected Values

#### Model Parameters

- $\text{OutputLength}$ = Number of output tokens generated
  Measured directly from model output. Must be a positive integer.

- $\text{TPS}$ = Tokens per second (generation speed)
  Available in model specifications or benchmarks. Large models typically generate 10-100 tokens/second.

- $\text{Latency}$ = Time to first token (seconds)
  Measured response time, usually between 0.1-2.0 seconds for most models.

#### Hardware Parameters (Reference values in Section 0)

- $P_{\text{GPU}}$ = Node-level **maximum** GPU power (kW)
  Specified in hardware documentation. Must align with the actual node configuration.

- $P_{\text{non-GPU}}$ = Node-level **maximum** power for non-GPU components (kW)
  From system specifications, typically 10-20% of GPU power draw.

- $\text{PUE}$ = Power Usage Effectiveness (data center multiplier)
  Data center efficiency metric, modern facilities range from 1.1-2.0.

#### Deployment Parameters

- $G$ = Number of GPUs assigned to the model
  Set in deployment configuration, cannot exceed available hardware.

- $N$ = Number of GPUs per node (e.g., 8 for DGX systems)
  Fixed by hardware configuration.

- $B$ = Batch size
  Configurable deployment parameter, typically 1-32 (default: 8).

#### Utilization Parameters

- $D_{\text{GPU}}$ = Per-GPU utilization fraction during inference
  Measured or provided by monitoring tools. Expect 5-15% for most workloads.

- $D_{\text{non-GPU}}$ = Per non-GPU utilization fraction
  Standard value of 0.5 (50%) based on paper findings.

- $`U_{\text{GPU\_total}}`$ = Aggregate GPU utilization fraction
  Calculated value, must be between 0 and 1.

- $`U_{\text{non-GPU\_total}}`$ = Aggregate non-GPU utilization fraction
  Calculated value, must be between 0 and 1.

---

## 4. Example — GPT-4o (Numbers Used in the Paper)

**GPT-4o Example Parameters**

- $P_{\text{GPU}} = 10.20$ kW (DGX H100/H200 node critical GPU power)
- $P_{\text{non-GPU}} = 1.6$ kW (node non-GPU peak)
- $\text{PUE} = 1.12$ (Azure)
- $G = 8$ (Large class → 8 GPUs assigned)
- $N = 8$ (GPUs per node)
- $B = 8$ (batch size)
- $D_{\text{non-GPU}} = 0.5$ (fixed)
- $D_{\text{GPU}} = 0.055 \ \text{to} \ 0.075$ (5.5%–7.5% per-GPU utilization for Large class on H100)

**Compute Utilizations**

```math
U_{\text{GPU\_total}} = \frac{8 \cdot (0.055 \ \text{to} \ 0.075)}{8 \cdot 8}
= 0.006875 \ \text{to} \ 0.009375
```

(0.6875% to 0.9375% of node GPU peak power)

```math
U_{\text{non-GPU\_total}} = \frac{8 \cdot 0.5}{8 \cdot 8} = 0.0625
```

(6.25% of node non-GPU peak power)

**Effective Powers**

```math
P_{\text{GPU,eff}} = 10.20 \cdot (0.006875 \ \text{to} \ 0.009375) = 0.0701 \ \text{kW} \ \text{to} \ 0.0956 \ \text{kW}
```

(70.1 W to 95.6 W)

```math
P_{\text{non-GPU,eff}} = 1.6 \cdot 0.0625 = 0.100 \ \text{kW}
```

(100 W)

```math
P_{\text{total,eff}} \approx 0.1701 \ \text{kW} \ \text{to} \ 0.1956 \ \text{kW}
```

(170.1 W to 195.6 W)

```math
P_{\text{withPUE}} = P_{\text{total,eff}} \cdot 1.12 \approx 0.1905 \ \text{kW} \ \text{to} \ 0.2191 \ \text{kW}
```

(190.5 W to 219.1 W)

**Per-Query Energy (Illustration for a Long Query)**  
Example long output length = 1500 tokens (paper’s long-form output).  
If $\text{time}_h \approx 0.0094$ h (≈ 33.84 s):

```math
E_{\text{query}} \approx 0.1905 \cdot 0.0094 \ \text{kWh} \approx 0.00179 \ \text{kWh} = 1.79 \ \text{Wh}
```

Matches the ≈ **1.788 Wh** per-query in the paper for long GPT-4o prompts.

## 5. References

1. Jegham, N., Abdelatti, M., Elmoubarki, L., Hendawi, A. "How Hungry is AI? A Systematic Methodology for Empirical Energy and Water Consumption Analysis of Large Language Models." arXiv:2505.09598 [cs.CY]. https://arxiv.org/abs/2505.09598

2. NVIDIA. NVIDIA Hopper GPUs Expand Reach as Demand for AI Grows.
   https://nvidianews.nvidia.com/news/nvidia-hopper-gpus-expand-reach-as-demand-for-ai-grows, March 2023. Accessed: 2025-04-2

3. Imran Latif, Alex C. Newkirk, Matthew R. Carbone, Arslan Munir, Yuewei Lin, Jonathan
   Koomey, Xi Yu, and Zhihua Dong. Single-node power demand during ai training: Measurements
   on an 8-gpu nvidia h100 system. IEEE Access, 13:61740–61747, 2025. doi: 10.1109/ACCESS.
   2025.3554728.

4. Noelle Walsh. How microsoft measures datacenter water and energy use to improve azure
   cloud sustainability. https://azure.microsoft.com/en-us/blog/how-microsoft-measures-datacenter-water-and-energy-use-to-improve-azure-cloud-sustainability/,
   April 2022. Microsoft Azure Blog.

5. NVIDIA Corporation. Nvidia dgx a100: The universal system for ai infrastructure. https://images.nvidia.com/aem-dam/Solutions/Data-Center/nvidia-dgx-a100-datasheet.pdf, 2020. Datasheet detailing specifications and features of the NVIDIA
   DGX A100 system.

---
