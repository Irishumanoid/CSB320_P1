import pandas as pd

def filter_zeros(dataset: pd.DataFrame, cols: list):
    print(f'cols {cols}')
    for name in cols:
        feature_median = dataset.describe()[name]['50%']
        for i in range(len(dataset[name])):
            if dataset.loc[i, name] == 0 or pd.isna(dataset.at[i, name]):
                dataset.loc[i, name] = feature_median
