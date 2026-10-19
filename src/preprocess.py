import re

import pandas as pd
from sklearn.model_selection import train_test_split

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

RANDOM_STATE = 42

def split_data(df):
    """Split into train (70%), validation (15%) and test (15%), stratified by category."""
    X = df["text"]
    y = df["category"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test