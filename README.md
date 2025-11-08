# Multi-Head-Attention-for-Textual-Entailment-NLI-
goal is to build a classifier that can determine the relationship between two sentences (a "premise" and a "hypothesis").


# Multi-Head Attention for Textual Entailment (SNLI) — README

## Overview

This project demonstrates basic data loading, inspection, and preprocessing for the Stanford Natural Language Inference (SNLI) dataset using PyTorch, HuggingFace Datasets, and Transformers. The workflow prepares the SNLI data for downstream natural language inference tasks, such as training models with multi-head attention.

---

## Steps Completed

### 1. Environment Setup and Imports

- Checked Python and PyTorch versions, and CUDA availability.
- Imported essential libraries:
  - `torch` for deep learning
  - `datasets` from HuggingFace for SNLI data
  - `transformers` for tokenization and data collation
  - `torch.utils.data` for data loading

### 2. Loading and Inspecting SNLI Data

- Loaded the SNLI dataset using `load_dataset("snli")`.
- Printed available splits (`train`, `validation`, `test`) and their sizes.
- Displayed the first 3 examples from the training split, showing:
  - Premise
  - Hypothesis
  - Label (0=entailment, 1=neutral, 2=contradiction, -1=missing)

### 3. Filtering and Label Distribution Analysis

- Filtered out examples with missing or invalid labels (`label is not None and label >= 0`).
- Used Python's `Counter` to count label distribution in each split.
- Printed label counts for train, validation, and test splits.

---

## Next Steps (Suggestions)

- Tokenize the premise and hypothesis sentences using a transformer tokenizer (e.g., BERT, RoBERTa).
- Prepare PyTorch DataLoaders for batching and training.
- Implement and train a multi-head attention model for textual entailment.
- Evaluate model performance on validation and test splits.

---

## Requirements

- Python 3.8+
- PyTorch
- transformers
- datasets

Install dependencies with:

```bash
pip install torch transformers datasets
```

---

## References

- [SNLI Dataset](https://nlp.stanford.edu/projects/snli/)
- [HuggingFace Datasets Documentation](https://huggingface.co/docs/datasets/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)