import pandas as pd
from models.synthesis_models import ctgan_synthesizer
from models.balancing_models import oversample_balance, undersample_balance, smote_balance, adasyn_balance

def anonymize(data, configs) -> pd.DataFrame:
    pass

def synthesize(data: pd.DataFrame, configs: dict) -> pd.DataFrame:
    synthesized_data = pd.DataFrame()
    if configs.get('model') == 'ctgan':
        synthesized_data = ctgan_synthesizer(data, configs)
    return synthesized_data

def balance(data: pd.DataFrame, configs: dict) -> pd.DataFrame:
    balanced_data = pd.DataFrame()
    if configs.get('model') == 'oversample':
        balanced_data = oversample_balance(data, configs)
    elif configs.get('model') == 'undersample':
        balanced_data = undersample_balance(data, configs)
    elif configs.get('model') == 'smote':
        balanced_data = smote_balance(data, configs)
    elif configs.get('model') == 'adasyn':
        balanced_data = adasyn_balance(data, configs)
    return balanced_data
    