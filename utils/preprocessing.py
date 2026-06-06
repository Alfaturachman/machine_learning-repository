import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2, random_state=42, stratify=None):
    test_ratio = 1 - train_ratio - val_ratio
    stratify_temp = stratify if stratify is not None else None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_ratio), random_state=random_state,
        stratify=stratify
    )
    val_size = val_ratio / (val_ratio + test_ratio)
    y_temp_stratify = y_temp if stratify is not None else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_size), random_state=random_state,
        stratify=y_temp_stratify
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def encode_categorical(df, columns, mapping=None):
    df = df.copy()
    for col in columns:
        if mapping and col in mapping:
            df[col] = df[col].map(mapping[col])
        else:
            df[col] = LabelEncoder().fit_transform(df[col])
    return df


def scale_features(X_train, X_test, X_val=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_test_scaled, X_val_scaled, scaler
    return X_train_scaled, X_test_scaled, scaler
