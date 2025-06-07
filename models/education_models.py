import pandas as pd
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from ctgan import CTGAN
import numpy as np # Import numpy for potential use within functions, even if not explicitly used in simple examples


def anonymize_education_data(dataframe):
    """
    Anonymizes 'name' and 'student_id' columns in a pandas DataFrame using Presidio.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing 'name' and 'student_id' columns.

    Returns:
    pd.DataFrame: A new DataFrame with 'anonymized_name' and 'anonymized_student_id' columns.

    Dependencies:
    - pandas
    - presidio-analyzer
    - presidio-anonymizer

    Expected Latency:
    Low to Medium - Processing time depends on the number of rows and the length of text in the 'name' and 'student_id' columns. For 1000 rows, it's expected to be relatively fast (seconds).

    API Specifications:
    - Endpoint: /anonymize
    - Method: POST
    - Request Body: JSON object with a 'data' key containing the DataFrame in a list of dictionaries format.
      Example: {"data": [{"student_id": "S1000", "name": "Riya Sharma", ...}, ...]}
    - Response Body: JSON object with a 'anonymized_data' key containing the DataFrame with anonymized columns in a list of dictionaries format.
      Example: {"anonymized_data": [{"anonymized_student_id": "<US_DRIVER_LICENSE>", "anonymized_name": "<PERSON>", ...}, ...]}
    """
    df_anonymized = dataframe.copy()

    # Initialize Presidio engines within the function to ensure they are available if the function is called independently
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    def anonymize_text(text):
        if pd.isna(text):
            return text
        text_str = str(text)
        results = analyzer.analyze(text=text_str, language='en')
        anonymized_text_result = anonymizer.anonymize(text_str, results)
        return anonymized_text_result.text

    df_anonymized['anonymized_name'] = df_anonymized['name'].apply(anonymize_text)
    df_anonymized['anonymized_student_id'] = df_anonymized['student_id'].apply(anonymize_text)
    # Drop the original columns to keep only anonymized versions
    df_anonymized.drop(columns=['name', 'student_id'], inplace=True)
    return df_anonymized

# Function 2: Balance Data
def balance_education_data(dataframe, target_column):
    """
    Balances the target_column in a pandas DataFrame using SMOTE.
    Handles missing values in 'gpa' by imputation (mean).
    One-hot encodes 'gender' and 'major'.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame.
    target_column (str): The name of the column to balance.

    Returns:
    pd.DataFrame: A new, balanced DataFrame containing numerical and one-hot encoded features,
                  and the balanced target column.

    Dependencies:
    - pandas
    - imblearn
    - sklearn
    - numpy

    Expected Latency:
    Medium - Processing involves feature engineering (imputation, one-hot encoding) and running the SMOTE algorithm, which scales with the number of samples and features. For 1000 rows, this should be manageable but takes longer than simple anonymization.

    API Specifications:
    - Endpoint: /balance
    - Method: POST
    - Request Body: JSON object with 'data' (DataFrame as list of dicts) and 'target_column' (string) keys.
      Example: {"data": [{"age": 23.0, "gpa": 2.98, ...}], "target_column": "passed_exam"}
    - Response Body: JSON object with a 'balanced_data' key containing the balanced DataFrame in a list of dictionaries format.
      Example: {"balanced_data": [{"age": 23.0, "gpa": 2.98, ... , "passed_exam": "Yes"}, ...]}
    """
    df_balanced = dataframe.copy()

    # Handle missing or invalid values in numerical columns
    numerical_columns = ['age', 'gpa', 'attendance', 'study_hours']
    for col in numerical_columns:
        if col in df_balanced.columns:
            # Replace empty strings with NaN
            df_balanced[col] = pd.to_numeric(df_balanced[col], errors='coerce')
            # Fill NaN values with the column mean
            df_balanced[col] = df_balanced[col].fillna(df_balanced[col].mean())

    # Select features (X) and target (y)
    feature_columns = ['age', 'gpa', 'attendance', 'study_hours', 'gender', 'major']
    X = df_balanced[feature_columns].copy()
    y = df_balanced[target_column].copy()

    # Convert categorical features to numerical using one-hot encoding
    X = pd.get_dummies(X, columns=['gender', 'major'], drop_first=True)

    # Encode the target variable to numerical
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Apply SMOTE for oversampling the minority class
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

    # Combine the resampled data back into a DataFrame
    X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled_df = pd.DataFrame(y_resampled, columns=[target_column])
    balanced_df_result = pd.concat([X_resampled_df, y_resampled_df], axis=1)

    # Decode the target variable back to original labels
    balanced_df_result[target_column] = le.inverse_transform(balanced_df_result[target_column])

    return balanced_df_result

# Function 3: Generate Synthetic Data
def generate_education_data(dataframe, num_samples, discrete_columns=None, epochs=10):
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