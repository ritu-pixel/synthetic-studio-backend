from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
import pandas as pd

def generate_metadata(df: pd.DataFrame, discrete_columns: list) -> Metadata:

    metadata = Metadata.detect_from_dataframe(df)
    for col in discrete_columns:
        metadata.update_column(
            column_name=col,
            sdtype='categorical'
        )
    return metadata

def ctgan_synthesizer(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    discrete_cols = config.get('discrete_columns', [])

    metadata = generate_metadata(df, discrete_cols)
    # print("Metadata generated:", metadata.to_dict())

    synthesizer = CTGANSynthesizer(
        metadata=metadata,
        epochs=config.get('epochs', 300),
        enforce_min_max_values=config.get('enforce_min_max_values', False),
        enforce_rounding=config.get('enforce_rounding', False)
    )

    synthesizer.fit(df)
    # print("Synthesizer fitted to data.")

    synthetic = synthesizer.sample(config.get('num_rows', 100))
    return synthetic

