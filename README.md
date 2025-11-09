# Multi-Head-Attention-for-Textual-Entailment-NLI-
goal is to build a classifier that can determine the relationship between two sentences (a "premise" and a "hypothesis").


## Overview

This project demonstrates basic data loading, inspection, and preprocessing for the Stanford Natural Language Inference (SNLI) dataset using PyTorch, HuggingFace Datasets, and Transformers. The workflow prepares the SNLI data for downstream natural language inference tasks, such as training models with multi-head attention.

---

## Steps Completed

## 1. Environment Setup and Imports

- Checked Python and PyTorch versions, and CUDA availability.
- Imported essential libraries:
  - `torch` for deep learning
  - `datasets` from HuggingFace for SNLI data
  - `transformers` for tokenization and data collation
  - `torch.utils.data` for data loading

## 2. Loading and Inspecting SNLI Data

- Loaded the SNLI dataset using `load_dataset("snli")`.
- Printed available splits (`train`, `validation`, `test`) and their sizes.
- Displayed the first 3 examples from the training split, showing:
  - Premise
  - Hypothesis
  - Label (0=entailment, 1=neutral, 2=contradiction, -1=missing)

## 3. Filtering and Label Distribution Analysis

- Filtered out examples with missing or invalid labels (`label is not None and label >= 0`).
- Used Python's `Counter` to count label distribution in each split.
- Printed label counts for train, validation, and test splits.

---

## Step 4: Tokenization of SNLI Dataset

In this step, the SNLI dataset was preprocessed for transformer-based models using the following procedure:

- **Tokenizer Initialization:**  
  Loaded the BERT tokenizer (`bert-base-uncased`) from HuggingFace Transformers with fast tokenization enabled.

- **Tokenizer Details Printed:**  
  Displayed the tokenizer class name, vocabulary size, and pad token information for reference.

- **Maximum Sequence Length Set:**  
  Chose a maximum token length of 64, suitable for the short sentences in SNLI.

- **Custom Tokenization Function:**  
  Defined a function to tokenize both the premise and hypothesis for each example, producing:
  - `premise_input_ids`
  - `premise_attention_mask`
  - `hypo_input_ids`
  - `hypo_attention_mask`

- **Batch Tokenization Applied:**  
  Used the `.map()` method to apply the tokenization function to all splits (`train`, `validation`, `test`) in the dataset, removing the original text columns to save space.

- **Result Verification:**  
  Printed the structure of the tokenized dataset and confirmed the new columns.

**Outcome:**  
The SNLI dataset is now fully tokenized and ready for use in transformer models, with each example containing token IDs and attention masks for both premise and hypothesis sentences.

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