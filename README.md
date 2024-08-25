# Enhanced Cognitive Distortions Detection and Classification through Data Augmentation Techniques
[![PRICAI-2024](http://img.shields.io/badge/PRICAI-2024-EF4444?style=for-the-badge&labelColor=8C2332)]()


This is the code for the PRICAI paper [Enhanced Cognitive Distortions Detection and Classification through Data Augmentation Techniques.]() 

By Mohamad Rasmy, [Caroline Sabty](https://scholar.google.com/citations?hl=en&user=EHOWWS8AAAAJ), [Nourhan Sakr](https://www.researchgate.net/profile/Nourhan-Sakr), and [Alia El Bolock](https://scholar.google.com/citations?user=APNJwoMAAAAJ&hl=en).

We present an enhanced approach for detecting and classifying cognitive distortions, building upon previous work by [Elsharawi et al. (2024)](https://aclanthology.org/2024.lrec-main.286/) that identified a CNN model using BERT embeddings as the most effective method. Our study explores additional embeddings from RoBERTa and GPT-2, implements fine-tuning of these models, and applies various data augmentation techniques to enhance the training dataset. The data augmentation techniques used in this study are:

- **Easy Data Augmentation (EDA):** This technique includes Synonym Replacement (SR), where random words in a sentence are replaced with their synonyms, and Random Insertion (RI), where a synonym of a random word is inserted at a random position in the sentence.

- **Word Embedding Substitution (WE_Sub):** This technique replaces words in a sentence with other words that have similar embeddings, providing greater flexibility and variation compared to synonym-based replacement.

- **Back-Translation (BT):** This technique involves translating a sentence from the source language to one or more intermediate languages and then back to the original language, introducing lexical and syntactic variations while preserving the overall semantic meaning.

We design our classification approach to address three distinct tasks:

- **Binary Classification (Distortion Detection):** Distinguishing between distorted and neutral data.

- **Multi-Class Classification (Distortion Classification):** Classifying distorted samples into fourteen cognitive distortion types.

- **Semantic Grouping Classification:** Grouping the fourteen distortion types into five semantic categories to address class imbalance and simplify the classification task.


# Usage

Each script can be run independently with specific command-line arguments.

### 1. Back Translation (back-translation.py)

This script performs back translation using Opus MT models.

```bash
python back-translation.py --data_path path/to/your/input.csv
```
Arguments:
- `--data_path`: Path to the input CSV file containing the data to be augmented.

### 2. Synonym Replacement and Random Insertion (sr-ri-eda.py)

This script implements Easy Data Augmentation (EDA) techniques: Synonym Replacement (SR) and Random Insertion (RI).

```bash
python sr-ri-eda.py --alpha 0.4 --total_augments_per_sentence 4 --data_path path/to/your/input.csv
```

Arguments:
- `--alpha`: Percentage of words to be changed in a sentence (float between 0 and 1).
- `--total_augments_per_sentence`: Total number of augmentations to generate per sentence.
- `--data_path`: Path to the input CSV file containing the data to be augmented.

### 3. Word Embedding Substitution using BERT (we-substitution-bert.py)

This script performs word substitution using BERT contextual embeddings.

```bash
python we-substitution-bert.py --alpha 0.3 --data_path path/to/your/input.csv
```
Arguments:
- `--alpha`: Percentage of words to be augmented in each sentence (float between 0 and 1).
- `--data_path`: Path to the input CSV file containing the data to be augmented.

### Note

- The input CSV file should contain a column named 'sentence' with the text data to be augmented.
- Each script will output a new CSV file with the augmented data in the same directory as the script.

# Citation
If you use this in your paper, please cite us:
```
[Citation will be available soon]
```

# Experiments

Experimental training for cognitive distortion detection can be found [here](https://github.com/MHRasmy/Enhanced-Cognitive-Distortions-Detection-and-Classification-through-Data-Augmentation-Techniques/tree/main/src/Experimental%20Training) for an experimental training used in the paper. Available hyperparameters at the start of the file to change training setups:

- `classification`: Set to 2/5/14 for different classification strategies.
- `val_pct`: Percentage of data for validation and test sets combined.
- `test_pct`: Percentage of test data from the `val_pct`. For example, if val_pct = 0.3 and test_pct = 2/3, then the test set would be (2/3)*0.3 = 0.2 and the actual validation set percentage would be the remaining (0.3-0.2) = 0.1.
- `model_name`: Set the transformer model name string (e.g., "bert-base-uncased", "roberta-base", "openai-community/gpt2").
- `is_CNN`: Boolean, if `true` would use CNN architecture with transformer `model_name` embeddings, if `false` would fine-tune the transformer model `model_name`.
- `api_key`: String set to your wandb API key.

