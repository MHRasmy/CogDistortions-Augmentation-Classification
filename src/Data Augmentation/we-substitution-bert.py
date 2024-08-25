# Install required dependencies
!pip install nlpaug --quiet
pip install --upgrade nlpaug torch --quiet

import argparse
import torch
import nlpaug.augmenter.word as naw
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Word Embedding Substitution using BERT")
parser.add_argument("--alpha", type=float, required=True, help="Percentage of word will be augmented")
parser.add_argument("--data_path", type=str, required=True, help="Path to the input CSV file")
args = parser.parse_args()


# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.device(device)

# Initialize the augmenter
aug = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased',
    action="substitute",
    stopwords=stopwords.words('english'),
    aug_p = args.alpha,
    device=device)


def substitute_words(sentence):
    augmented_sentence = aug.augment(sentence)
    return augmented_sentence


# Read the dataset
df = pd.read_csv(args.data_path)

# Augment the dataset
df['augmented_sentence'] = df['sentence'].map(substitute_words)

# Save the augmented dataset
df.to_csv("DA-Substitution-BERT.csv", index=False)

