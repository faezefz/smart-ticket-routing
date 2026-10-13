import re

import pandas as pd

def clean_text(text):
    """Normalizes a customer message."""
    text = text.lower()
    text = re.sub(r"\{\{.*?\}\}", "<entity>", text)  # {{...}} → <entity>
    text = re.sub(r"\d+", "<entity>", text)  # numbers → <entity>
    text = re.sub(r"\s+", " ", text)  # extra spaces → one space
    return text.strip()

def prepare_data(df):
    """Clean messages and remove duplicates."""
    df = df[["instruction", "category"]].copy()
    df["text"] = df["instruction"].apply(clean_text)
    df = df.drop_duplicates(subset=["text", "category"])
    return df