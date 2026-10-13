from pathlib import Path
import pandas as pd
from datasets import load_dataset

DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
RAW_PATH = Path("data/raw.csv")


def download_data(path=RAW_PATH):
    """Download the dataset from Hugging Face and save it as a CSV file."""
    dataset = load_dataset(DATASET_NAME, split="train")
    df = dataset.to_pandas()
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    df = download_data()
    print(df.shape)
    