import pandas as pd

FEATURES = ['year', 'make', 'model', 'trim', 'body', 'transmission', 'state', 'condition', 'odometer', 'color', 'interior']
CAT_FEATURES = ['make', 'model', 'trim', 'body', 'transmission', 'state', 'color', 'interior']
NUM_FEATURES = ['year', 'condition', 'odometer']

def load_and_preprocess(path: str, nrows=100000):
    df = pd.read_csv(path, nrows=nrows).dropna(subset=['sellingprice'])
    df[CAT_FEATURES] = df[CAT_FEATURES].fillna('Unknown').astype(str)
    for col in NUM_FEATURES:
        df[col] = df[col].fillna(df[col].median())
    return df[FEATURES], df['sellingprice']