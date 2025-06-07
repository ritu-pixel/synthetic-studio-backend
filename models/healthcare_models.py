import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from ctgan import CTGAN


def anonymize_health_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Anonymizes the 'Patient_ID' column in a pandas DataFrame.

    Args:
        df: The input pandas DataFrame.

    Returns:
        A DataFrame with the 'Patient_ID' column anonymized.
    """
    # Create a copy to avoid modifying the original DataFrame outside the function scope
    df_anonymized = df.copy()

    # Replace each Patient_ID with a placeholder
    df_anonymized['Patient_ID'] = '[PATIENT_ID]'

    return df_anonymized

def generate_health_data(dataframe, num_samples, discrete_columns=None, epochs=10):
    """
    Generates synthetic data from a pandas DataFrame using CTGAN.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame to train the CTGAN model on.
    num_samples (int): The desired number of synthetic samples to generate.

    Returns:
    pd.DataFrame: A DataFrame containing the generated synthetic samples.

    Dependencies:
    - pandas
    - ctgan

    Expected Latency:
    High - Training the CTGAN model is computationally intensive and can take significant time, especially for larger datasets or more epochs. Generating samples is relatively faster but still contributes to the overall latency.

    API Specifications:
    - Endpoint: /synthesize
    - Method: POST
    - Request Body: JSON object with 'data' (DataFrame as list of dicts) and 'num_samples' (integer) keys.
      Example: {"data": [{"age": 23.0, "gpa": 2.98, ...}], "num_samples": 2000}
    - Response Body: JSON object with a 'synthetic_data' key containing the generated DataFrame in a list of dictionaries format.
      Example: {"synthetic_data": [{"age": 21.5, "gpa": 3.1, ...}, ...]}
    """
    # Identify discrete columns
    if discrete_columns is None:
        # Automatically detect discrete columns based on dtype
        discrete_columns = [col for col in dataframe.columns if dataframe[col].dtype == 'bool' or dataframe[col].dtype == 'object']

    # Instantiate the CTGAN synthesizer
    ctgan = CTGAN(verbose=False)

    # Fit the synthesizer to the DataFrame
    # Using a small number of epochs for demonstration purposes within a function
    ctgan.fit(dataframe, discrete_columns=discrete_columns, epochs=epochs)

    # Generate synthetic data
    synthetic_data_result = ctgan.sample(num_samples)

    return synthetic_data_result


def balance_health_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Balances the dataset using SMOTE on numerical features.

    Assumes the input DataFrame has a 'Diagnosis' column as the target variable.
    Categorical features in the input DataFrame are dropped before applying SMOTE.

    Args:
        df: The input pandas DataFrame with a 'Diagnosis' column.

    Returns:
        A balanced pandas DataFrame with numerical features and the 'Diagnosis' column.
    """
    df_cleaned = df.dropna(subset=['Diagnosis']).copy()

    X = df_cleaned.drop('Diagnosis', axis=1)
    y = df_cleaned['Diagnosis']
    X_numerical = X.select_dtypes(include=['number'])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Calculate the size of the smallest class
    class_counts = pd.Series(y_encoded).value_counts()
    min_class_size = class_counts.min()

    if min_class_size < 2:
        raise ValueError(f"SMOTE requires at least 2 samples in the minority class, but found {min_class_size}")

    # Adjust k_neighbors to avoid the ValueError
    k_neighbors = min(5, min_class_size - 1)

    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_resampled, y_resampled_encoded = smote.fit_resample(X_numerical, y_encoded)

    y_resampled = label_encoder.inverse_transform(y_resampled_encoded)
    balanced_df = pd.DataFrame(X_resampled, columns=X_numerical.columns)
    balanced_df['Diagnosis'] = y_resampled

    return balanced_df
