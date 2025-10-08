# utils/data_utils.py
import pandas as pd
from typing import Tuple, Union, Dict, Any

# Place the read_prism_csv function here
def read_prism_csv(path):
    # ... (your full function code) ...
    df = pd.read_csv(path, index_col=False)
    df_header = df.iloc[:4,:]
    df.columns = df.iloc[1].values
    df = df.iloc[4:,:]
    df.rename(columns={'Extended Name':'DATETIME'}, inplace=True)
    df.reset_index(drop=True, inplace=True)
    # Convert to numeric, coercing errors, but keep DATETIME as datetime
    numeric_cols = [col for col in df.columns if col != 'DATETIME']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    return df, df_header

# Place the split_holdout function here
def split_holdout(
    cleaned_df: pd.DataFrame,
    split_mark: Union[float, str, pd.Timestamp],
    date_col: str = "DATETIME",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, Dict[str, Dict[str, Any]]]:
    # ... (your full function code, modified slightly to accept a DataFrame directly) ...
    # NOTE: I've simplified this to take a DataFrame directly, which is better for Streamlit
    
    cleaned_df[date_col] = pd.to_datetime(cleaned_df[date_col])
    cleaned_df_sorted = cleaned_df.sort_values(date_col).reset_index(drop=True)

    if isinstance(split_mark, float):
        n = len(cleaned_df_sorted)
        h_size = int(round(n * split_mark))
        split_idx = n - h_size - 1 if h_size < n else n - 1
        split_mark = cleaned_df_sorted.iloc[split_idx][date_col]
    else:
        split_mark = pd.to_datetime(split_mark)

    train_val_df = cleaned_df_sorted[cleaned_df_sorted[date_col] <= split_mark].reset_index(drop=True)
    holdout_df = cleaned_df_sorted[cleaned_df_sorted[date_col] > split_mark].reset_index(drop=True)

    def get_stats(df):
        if len(df) > 0:
            start = df[date_col].iloc[0]
            end = df[date_col].iloc[-1]
            num_days = (end - start).days + 1
            return {"start": start, "end": end, "size": len(df), "num_days": num_days}
        return {"start": None, "end": None, "size": 0, "num_days": 0}

    stats = {
        "cleaned": get_stats(cleaned_df_sorted),
        "train_val": get_stats(train_val_df),
        "holdout": get_stats(holdout_df),
    }

    return train_val_df, holdout_df, split_mark, stats