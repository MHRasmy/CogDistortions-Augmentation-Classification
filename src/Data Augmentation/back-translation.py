# Installing required dependencies
!pip install torch transformers datasets sacremoses sentencepiece pandas==1.5.3 --quiet
!pip install ctranslate2


# Downloading the Opus-MT models for English-German, German-French, and French-English translation
!ct2-transformers-converter --model Helsinki-NLP/opus-mt-en-de --output_dir opus-mt-en-de
!ct2-transformers-converter --model Helsinki-NLP/opus-mt-de-fr --output_dir opus-mt-de-fr
!ct2-transformers-converter --model Helsinki-NLP/opus-mt-fr-en --output_dir opus-mt-fr-en

import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import Dataset
import ctranslate2
import transformers
import time


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Back Translation Augmentation using Opus MT")
parser.add_argument("--data_path", type=str, required=True, help="Path to the input CSV file")
args = parser.parse_args()

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load models and tokenizers
translator1 = ctranslate2.Translator("opus-mt-en-de", device='cuda')
tokenizer1 = transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")

translator2 = ctranslate2.Translator("opus-mt-de-fr", device='cuda')
tokenizer2 = transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-fr")

translator3 = ctranslate2.Translator("opus-mt-fr-en", device='cuda')
tokenizer3 = transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

df['BT_Opus_MT'] = np.nan

def back_translation(data_batch):
    with torch.no_grad():
        text_to_aug = data_batch["sentence"]
        
        # Translate original sentences from English to German
        tokenized_sentences1 = [tokenizer1.convert_ids_to_tokens(tokenizer1.encode(sentence)) for sentence in text_to_aug]
        results1 = translator1.translate_batch(tokenized_sentences1)
        german_translation = [tokenizer1.decode(tokenizer1.convert_tokens_to_ids(result.hypotheses[0])) for result in results1]

        
        # Translate German translations to French
        tokenized_sentences2 = [tokenizer2.convert_ids_to_tokens(tokenizer2.encode(sentence)) for sentence in german_translation]
        results2 = translator2.translate_batch(tokenized_sentences2)
        french_translation = [tokenizer2.decode(tokenizer2.convert_tokens_to_ids(result.hypotheses[0])) for result in results2]

        
        # Translate French translations back to English
        tokenized_sentences3 = [tokenizer3.convert_ids_to_tokens(tokenizer3.encode(sentence)) for sentence in french_translation]
        results3 = translator3.translate_batch(tokenized_sentences3)
        back_translated_sentences = [tokenizer3.decode(tokenizer3.convert_tokens_to_ids(result.hypotheses[0])) for result in results3]

        
        data_batch["BT_Opus_MT"] = back_translated_sentences

    return data_batch


# Main execution
if __name__ == "__main__":
    try:
        # Read the dataset
        df = pd.read_csv(args.data_path)

        # Initialize the 'BT_Opus_MT' column with NaN values
        df['BT_Opus_MT'] = np.nan
        
        # Convert to Hugging Face dataset
        dataset = Dataset.from_pandas(df)
        
        # Perform back translation
        back_translated_datasets = dataset.map(back_translation, batched=True)
        
        # Save the augmented dataset
        back_translated_datasets.to_csv("Back-Translation-Augmented.csv", index=False)
        
        print("Back translation completed successfully. Output saved as 'Back-Translation-Augmented.csv'.")
    
    except FileNotFoundError:
        print(f"Error: The file {args.data_path} was not found. Please check the path and try again.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file {args.data_path} is empty. Please check the file and try again.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

