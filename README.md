# LLM Autoimmune Symptom Fit
## Probing disease-symptom associations with surprisal and classification

### Overview

Large language models trained on large text corpora appear to acquire structured knowledge about the world. In health-related contexts, an important question is whether these models encode medically meaningful relationships such as disease-symptom associations.

This project evaluates whether transformer language models assign lower surprisal to symptoms that are clinically associated with a given autoimmune disease compared to real but mismatched symptoms from other autoimmune conditions.

The project combines two analyses:

1. **Surprisal analysis**
   - Measure how expected a symptom is given a disease context

2. **Classification**
   - Use surprisal features to predict whether a disease-symptom pair is clinically matched

---

### Research Question

Do language models assign lower surprisal to symptoms that are clinically associated with a given autoimmune disease than to symptoms that belong to other autoimmune diseases?

---

### Why this matters

Language models are increasingly used in health-adjacent applications such as symptom search, triage assistants, and medical documentation tools.

Understanding whether models encode disease-symptom associations helps evaluate:

- reliability of biomedical knowledge in LLMs
- risks of hallucinated or mismatched medical information
- potential use of LLM probabilities as signals for knowledge probing

---

### Dataset

The dataset contains autoimmune diseases and symptoms categorized as:

- **matched** – symptom associated with the disease
- **mismatched** – symptom from another autoimmune disease

Example:

| disease | symptom | label |
|-------|-------|------|
| Lupus | butterfly rash | 1 |
| Lupus | dry eyes | 0 |
| Sjogren's syndrome | dry eyes | 1 |
| Sjogren's syndrome | joint swelling | 0 |

---

### Method

For each disease-symptom pair we generate templated sentences such as:

> A patient with lupus often experiences butterfly rash.

We then compute the **surprisal of the symptom token**:

surprisal = −log₂ P(symptom | disease context)

Lower surprisal indicates that the model considers the symptom more expected.

---

### Models

- GPT-2
- GPT-2 Medium
- GPT-2 Large

---

### Analyses

1. Compare surprisal distributions for matched vs mismatched symptoms
2. Paired statistical testing
3. Train a classifier using surprisal features to predict symptom correctness

---

### Results

The project evaluates whether:

- matched symptoms receive lower surprisal
- model size improves disease-symptom discrimination
- surprisal can be used as a signal for classification

---

### Future Work

- Extend to neurological disorders
- Compare newer open-weight models
- Incorporate clinical ontologies such as UMLS
- Evaluate models on real clinical text

---

### Running the Project
