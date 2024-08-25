# This code is inspired by and adapted from the "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks" Paper by Jason Wei, Kai Zou
# Original source: https://github.com/jasonwei20/eda_nlp

# Installing required dependencies 
pip install transformers nltk

import argparse
import nltk
import random
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Easy Data Augmentation (EDA) using Synonym Replacement (SR) and Random Insertion (RI)")
parser.add_argument("--alpha", type=float, required=True, help="Percentage of words to be changed in a sentence")
parser.add_argument("--total_augments_per_sentence", type=int, required=True, help="Total number of augmentations per sentence")
parser.add_argument("--data_path", type=str, required=True, help="Path to the input CSV file")
args = parser.parse_args()

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.device(device)


# Function definitions (synonym_replacement, random_insertion, get_synonyms, add_word)

def synonym_replacement(sentence, n):
    """
    Perform synonym replacement in a sentence.

    Parameters:
    sentence (str): The original sentence.
    n (int): The number of words in the sentence to replace.

    Returns:
    str: The augmented sentence with n words replaced by their synonyms.
    """
    words = sentence.split()
    new_words = words.copy()
    word_replacement_count = {}  # Track the number of times each word is replaced

    # Get shuffled list of words that have synonyms
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0

    for random_word in random_word_list:
        # Get a list of synonyms with decreasing similarity
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            # Determine how many times this word has already been replaced
            times_replaced = word_replacement_count.get(random_word, 0)

            # Select a synonym with decreasing similarity
            if times_replaced < len(synonyms):
                synonym = synonyms[times_replaced]
                # Replace the word with its synonym
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
                word_replacement_count[random_word] = times_replaced + 1
            else:
                continue
        if num_replaced >= n:
            break

    sentence = ' '.join(new_words)
    return sentence


def random_insertion(sentence, n):
    """
    Perform random insertion in a sentence.

    Parameters:
    sentence (str): The original sentence.
    n (int): The number of random insertions to perform.

    Returns:
    str: The augmented sentence with n random insertions.
    """
    words = sentence.split()
    new_words = words.copy()

    for _ in range(n):
        # Insert a random synonym of a word
        add_word(new_words)

    sentence = ' '.join(new_words)
    return sentence


def get_synonyms(word):
    """
    Get synonyms of a word.

    Parameters:
    word (str): The word for which to find synonyms.

    Returns:
    list: A list of synonyms for the word.
    """
    synonyms = set()
    # Get all synonyms sets (synsets) for the word
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char.isalpha()])
            synonyms.add(synonym)
    # Remove the word itself from its synonym list, if present
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


def add_word(new_words):
    """
    Add a random synonym of a random word into a list of words.

    Parameters:
    new_words (list): A list of words representing a sentence.

    """
    counter = 0
    while True:
        # Select a random word from the sentence
        random_word = new_words[random.randint(0, len(new_words)-1)]
        # Get synonyms for the selected word
        synonyms = get_synonyms(random_word)
        # Break the loop if no synonyms found after several tries
        if counter >= 10 or len(synonyms) > 0:
            break
        counter += 1
    # If synonyms found, insert a random synonym at a random position
    if synonyms:
        random_synonym = random.choice(synonyms)
        random_idx = random.randint(0, len(new_words)-1)
        new_words.insert(random_idx, random_synonym)

# ## Applying the augmentation to the dataset

def augment_sentence(sentence, total_augments, alpha):
    words = sentence.split()
    l = len(words)
    # Calculate n, ensure at least 1 word is changed
    n = max(1, int(alpha * l)) 
    sr_count = ri_count = total_augments // 2
    augmented_sentences = []
    for _ in range(sr_count):
        augmented_sentences.append(synonym_replacement(sentence, n))
    for _ in range(ri_count):
        augmented_sentences.append(random_insertion(sentence, n))

    return augmented_sentences


# Read the dataset
df = pd.read_csv(args.data_path)

# Adding the augmented sentences to new column
df['augmented_sentences'] = df['sentence'].apply(lambda x: augment_sentence(x, args.total_augments_per_sentence, args.alpha))

# Saving the augmented dataset 
df.to_csv("EDA-SR-RI.csv", index=False)