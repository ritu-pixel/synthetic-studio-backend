import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from sdv.single_table.ctgan import CTGANSynthesizer
from sklearn.preprocessing import LabelEncoder
from sdv.metadata import SingleTableMetadata
import hashlib
import spacy

nlp = spacy.load("en_core_web_sm")

def anonymize_names_spacy(text_series: pd.Series) -> pd.Series:
    anonymized_texts = []
    name_map = {}
    current_id = 1
    for text in text_series:
        if not isinstance(text, str):
            anonymized_texts.append(text)
            continue
        doc = nlp(text)
        anonymized_name = text
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                original_name = ent.text
                if original_name not in name_map:
                    name_map[original_name] = f"NAME_{current_id}"
                    current_id += 1
                anonymized_name = anonymized_name.replace(original_name, name_map[original_name])
        anonymized_texts.append(anonymized_name)
    return pd.Series(anonymized_texts)

def anonymize_finance_data(df: pd.DataFrame) -> pd.DataFrame:
    # Anonymize customer_id
    df['customer_id'] = ['CUSTOMER_' + str(i + 1) for i in range(len(df))]

    # Anonymize names
    df['name'] = anonymize_names_spacy(df['name'])

    # Ensure 'age' column is numeric
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Generalize age into bins
    bins = [0, 30, 45, 60, 75, 100]
    labels = ['30', '30-44', '45-59', '60-74', '75+']
    df['age_binned'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    df.drop(columns=['age'], inplace=True)

    # Ensure 'income' column is numeric
    df['income'] = pd.to_numeric(df['income'], errors='coerce')

    # Generalize income into bins
    income_bins = [0, 3000, 6000, 10000, 15000, 1000000]
    income_labels = ['Low', 'Medium', 'High', 'Very High', 'Ultra']
    df['income_binned'] = pd.cut(df['income'].fillna(0), bins=income_bins, labels=income_labels, right=False)
    df.drop(columns=['income'], inplace=True)

    return df

def generate_finance_data(df, num_rows, categorical_cols, epochs) -> pd.DataFrame:
    # Create and detect metadata from the real dataframe
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)

    if categorical_cols is None:
        categorical_cols = ['gender', 'employment_type', 'has_defaulted', 'name']

    # Mark all categorical columns explicitly
    for col in categorical_cols:
        metadata.update_column(column_name=col, sdtype='categorical')

    # Initialize CTGANSynthesizer with metadata
    model = CTGANSynthesizer(metadata=metadata, epochs=epochs)

    # Fit and generate synthetic samples
    model.fit(df)
    synthetic_df = model.sample(num_rows)

    return synthetic_df

def balance_finance_data(df, target_col):
    df_copy = df.copy()

    # Drop non-numeric ID-like columns that are not features
    if 'customer_id' in df_copy.columns:
        df_copy.drop(columns=['customer_id'], inplace=True)

    # Ensure all numeric columns are valid
    for col in df_copy.select_dtypes(include=['object', 'string']).columns:
        try:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        except Exception:
            pass

    # Drop rows with NaN values in the target column
    df_copy.dropna(subset=[target_col], inplace=True)

    # Fill or drop NaN values in the feature columns
    df_copy.fillna(0, inplace=True)  # Replace NaN with 0 (or use another strategy)

    X = df_copy.drop(columns=[target_col])
    y = df_copy[target_col]

    # Ensure categorical columns are of type string
    categorical_cols = ['gender', 'employment_type', 'has_defaulted', 'age_binned', 'income_binned', 'name']
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype(str)

    # Find categorical feature indices
    categorical_features = [X.columns.get_loc(col) for col in categorical_cols if col in X.columns]

    smote = SMOTENC(categorical_features=categorical_features, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
    df_balanced[target_col] = y_resampled
    return df_balanced