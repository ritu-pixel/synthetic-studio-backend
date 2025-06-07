import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

def oversample_balance(data: pd.DataFrame, configs: dict) -> pd.DataFrame:
    ros = RandomOverSampler(random_state=configs.get('random_state', 42))
    x, y = data.drop(columns=configs['target']), data[configs['target']]
    x_resampled, y_resampled = ros.fit_resample(x, y)
    return pd.concat([x_resampled, y_resampled], axis=1)

def undersample_balance(data: pd.DataFrame, configs: dict) -> pd.DataFrame:
    rus = RandomUnderSampler(random_state=configs.get('random_state', 42))
    x, y = data.drop(columns=configs['target']), data[configs['target']]
    x_resampled, y_resampled = rus.fit_resample(x, y)
    return pd.concat([x_resampled, y_resampled], axis=1)

def smote_balance(data: pd.DataFrame, configs: dict) -> pd.DataFrame:
    smote = SMOTE(random_state=configs.get('random_state', 42))
    x, y = data.drop(columns=configs['target']), data[configs['target']]
    x_resampled, y_resampled = smote.fit_resample(x, y)
    return pd.concat([x_resampled, y_resampled], axis=1)

def adasyn_balance(data: pd.DataFrame, configs: dict) -> pd.DataFrame:
    adasyn = ADASYN(random_state=configs.get('random_state', 42))
    x, y = data.drop(columns=configs['target']), data[configs['target']]
    x_resampled, y_resampled = adasyn.fit_resample(x, y)
    return pd.concat([x_resampled, y_resampled], axis=1)