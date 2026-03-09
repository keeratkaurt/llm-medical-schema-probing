# LLM Medical Schema Probing

A research-style evaluation of whether general-purpose language models encode clinically plausible symptom schemas.

## Overview

Large language models are trained on human-generated text and often appear to acquire structured knowledge about the world. In high-stakes settings like healthcare, an important question is whether these models distinguish biologically plausible symptom descriptions from impossible or nonsensical ones.

This project probes that question using a controlled surprisal-based paradigm. I constructed minimal sentence pairs describing autoimmune diseases, where each pair ends in either:

- a clinically plausible symptom, or
- a biologically impossible symptom.

I then measured the surprisal of the final symptom token under GPT-2, GPT-2 Medium, and GPT-2 Large. If a model has internalized disease-related semantic expectations, plausible symptom endings should be more predictable and therefore have lower surprisal.

## Research Question

Do language models assign lower surprisal to clinically plausible autoimmune symptoms than to biologically impossible ones, suggesting the presence of emergent medical schema-like knowledge?

## Why this matters

Language models are increasingly used in health-adjacent contexts such as medical search, symptom triage, documentation support, and question answering. Even when they are not clinically deployed, they may still influence how users reason about symptoms and disease.

This project is relevant to neurotech and health AI because it evaluates whether model behavior reflects structured semantic expectations in a medically meaningful domain, while also testing an important limitation: whether apparent “medical reasoning” may instead be driven by lexical rarity.

## Dataset

The dataset contains 12 minimal pairs (24 total sentences). Each item names an autoimmune disease and ends with either:

- a plausible symptom (for example fatigue, rash, joint pain), or
- an implausible symptom (for example levitation, glowing fingernails, extra limbs).

The sentence structure is kept as parallel as possible so the final symptom drives the manipulation.

## Models

- GPT-2
- GPT-2 Medium
- GPT-2 Large

## Method

For each sentence:
1. Tokenize the input with the model tokenizer
2. Identify the final token
3. Compute surprisal as:

   surprisal = -log2 p(token | preceding context)

4. Compare surprisal across plausible and implausible conditions
5. Visualize differences across models

## Main Finding

Across all three models, implausible symptom endings receive higher surprisal than plausible symptom endings. Larger models show a stronger separation, suggesting more structured expectations about symptom plausibility.

## Important Limitation

This effect may not reflect purely medical schema knowledge. Some implausible endings may also be rarer lexical items, which can independently increase surprisal. The notebook therefore includes lexical-confound checks and outlines next-step controls for strengthening causal interpretation.

## Repository Structure

```text
data/        stimulus set
notebooks/   main analysis notebook
results/     exported tables and figures
src/         reusable utility code
