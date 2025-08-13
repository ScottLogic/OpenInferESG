# Methodology: Calculating Per-Query Energy for LLM Inference

This methodology is based on the research paper by Jegham et al. (2025) *"How Hungry is AI? A Systematic Methodology for Empirical Energy and Water Consumption Analysis of Large Language Models"* [1] which presents a framework for estimating the energy consumption of Large Language Models (LLMs) during inference in data center environments. The approach provides a systematic way to calculate per-query energy usage, taking into account various hardware and deployment factors. Section 0 provides current hardware specifications for major cloud LLM deployments, which can be used as reference values in the calculations.

> To view this as a markdown preview in VSCode press `Ctrl+Shift+V`

## Table of Contents
- [Introduction](#introduction)
- [0. Hardware Specifications](#0-cloud-hardware-specifications-for-openai-models)
- [1. Core Concept](#1-core-concept)
- [2. Core Equations and Variables](#2-core-equations-and-variables)
- [3. Utilization Model](#3-utilization-model)
- [4. Practical Example: GPT-4o](#4-practical-example-gpt-4o)
- [5. Practical Considerations](#5-practical-considerations)
- [6. References](#6-references)

## Introduction

This methodology is based on the research paper by Jegham et al. (2025) *"How Hungry is AI? A Systematic Methodology for Empirical Energy and Water Consumption Analysis of Large Language Models"* [1] which presents a framework for estimating the energy consumption of Large Language Models (LLMs) during inference in data center environments. The approach provides a systematic way to calculate per-query energy usage, taking into account various hardware and deployment factors. Section 0 provides current hardware specifications for major cloud LLM deployments, which can be used as reference values in the calculations.

## 0. Cloud Hardware Specifications for OpenAI Models

This section provides reference values for hardware configurations used in major cloud LLM deployments. These specifications can be used as input parameters for the energy calculations detailed in the following sections.

### 0.1 Deployment Specifications

**Table 1 – Deployment and Infrastructure Specifications (OpenAI Models)**

| Models                                                                                                   | Launch Dates        | Host            | Hardware            | Critical Power (kW) | PUE  |
|----------------------------------------------------------------------------------------------------------|---------------------|-----------------|---------------------|---------------------|------|
| GPT-4.1, GPT-4.1 mini, GPT-4.1 nano, o4-mini (high), GPT-4.5, o3, o3-mini (high), o3-mini, o1, o1-mini, GPT-4o (Mar ’25) | Apr 2025 – May 2024 | Microsoft Azure | DGX H200/H100       | 10.20               | 1.12 |
| GPT-4o mini, GPT-4 Turbo, GPT-4                                                                           | Jul 2024 – Mar 2023 | Microsoft Azure | DGX A100*           | 6.50                | 1.12 |


## 1. Core Concept

The methodology estimates energy consumption per query through three key steps:
1. Calculate the inference time for the request (in hours)
2. Multiply by the **effective system power** (combining GPU and non-GPU components)
3. Scale by the data center's Power Usage Effectiveness (PUE)

## 2. Core Equations and Variables

### 2.1 Energy Consumption Formula

The core equation for calculating per-query energy consumption is:

$$
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
$$

### 2.2 Utilization Formulas

The utilization components are calculated as:

$$
U_{\text{GPU\_total}} = \frac{G \cdot D_{\text{GPU}}}{N \cdot B}
\qquad,\qquad
U_{\text{non-GPU\_total}} = \frac{G \cdot D_{\text{non-GPU}}}{N \cdot B}
$$

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

- $U_{\text{GPU\_total}}$ = Aggregate GPU utilization fraction
  Calculated value, must be between 0 and 1.

- $U_{\text{non-GPU\_total}}$ = Aggregate non-GPU utilization fraction
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

$$
U_{\text{GPU\_total}} = \frac{8 \cdot (0.055 \ \text{to} \ 0.075)}{8 \cdot 8}
= 0.006875 \ \text{to} \ 0.009375
$$
(0.6875% to 0.9375% of node GPU peak power)

$$
U_{\text{non-GPU\_total}} = \frac{8 \cdot 0.5}{8 \cdot 8} = 0.0625
$$
(6.25% of node non-GPU peak power)

**Effective Powers**

$$
P_{\text{GPU,eff}} = 10.20 \cdot (0.006875 \ \text{to} \ 0.009375) = 0.0701 \ \text{kW} \ \text{to} \ 0.0956 \ \text{kW}
$$
(70.1 W to 95.6 W)

$$
P_{\text{non-GPU,eff}} = 1.6 \cdot 0.0625 = 0.100 \ \text{kW}
$$
(100 W)

$$
P_{\text{total,eff}} \approx 0.1701 \ \text{kW} \ \text{to} \ 0.1956 \ \text{kW}
$$
(170.1 W to 195.6 W)

$$
P_{\text{withPUE}} = P_{\text{total,eff}} \cdot 1.12 \approx 0.1905 \ \text{kW} \ \text{to} \ 0.2191 \ \text{kW}
$$
(190.5 W to 219.1 W)

**Per-Query Energy (Illustration for a Long Query)**  
Example long output length = 1500 tokens (paper’s long-form output).  
If $\text{time}_h \approx 0.0094$ h (≈ 33.84 s):

$$
E_{\text{query}} \approx 0.1905 \cdot 0.0094 \ \text{kWh} \approx 0.00179 \ \text{kWh} = 1.79 \ \text{Wh}
$$

Matches the ≈ **1.788 Wh** per-query in the paper for long GPT-4o prompts.

## 5. Practical Considerations

When implementing this methodology, consider:

1. **Data Availability**
   - Some parameters may require estimation or benchmarking
   - Actual GPU utilization may vary based on implementation

2. **System Variations**
   - Different hardware configurations will affect calculations
   - Cloud providers may have different PUE values

3. **Workload Patterns**
   - Batch processing may affect real-world energy consumption
   - Consider peak vs. average utilization

4. **Monitoring and Validation**
   - Implement monitoring to validate calculations
   - Regularly update parameters based on real measurements

## 6. References

1. Jegham, N., Abdelatti, M., Elmoubarki, L., Hendawi, A. "How Hungry is AI? A Systematic Methodology for Empirical Energy and Water Consumption Analysis of Large Language Models." arXiv:2505.09598 [cs.CY]. https://arxiv.org/abs/2505.09598

---
