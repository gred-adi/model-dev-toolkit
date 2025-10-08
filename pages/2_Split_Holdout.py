import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import datetime
import os
import tkinter as tk
from tkinter import filedialog
import re

# This utility function would be in your utils/data_utils.py file
# For this self-contained example, it is assumed to exist.
# from utils.data_utils import split_holdout
# --- Placeholder for the split_holdout function from utils/data_utils.py ---
from typing import Tuple, Union, Dict, Any
def split_holdout(
    cleaned_df: pd.DataFrame,
    split_mark: Union[float, str, pd.Timestamp],
    date_col: str = "Point Name",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, Dict[str, Dict[str, Any]]]:
    cleaned_df[date_col] = pd.to_datetime(cleaned_df[date_col])
    cleaned_df_sorted = cleaned_df.sort_values(date_col).reset_index(drop=True)

    if isinstance(split_mark, float):
        n = len(cleaned_df_sorted)
        h_size = int(round(n * split_mark))
        split_idx = n - h_size - 1 if h_size < n else n - 1
        split_timestamp = cleaned_df_sorted.iloc[split_idx][date_col]
    else:
        split_timestamp = pd.to_datetime(split_mark)

    train_val_df = cleaned_df_sorted[cleaned_df_sorted[date_col] <= split_timestamp].reset_index(drop=True)
    holdout_df = cleaned_df_sorted[cleaned_df_sorted[date_col] > split_timestamp].reset_index(drop=True)

    def get_stats_local(df):
        if len(df) > 0:
            start = df[date_col].iloc[0]
            end = df[date_col].iloc[-1]
            return {"start": start, "end": end, "size": len(df)}
        return {"start": None, "end": None, "size": 0}

    stats = {
        "cleaned": get_stats_local(cleaned_df_sorted),
        "train_val": get_stats_local(train_val_df),
        "holdout": get_stats_local(holdout_df),
    }
    return train_val_df, holdout_df, split_timestamp, stats
# --- End of placeholder function ---


st.set_page_config(page_title="Data Splitting", layout="wide", initial_sidebar_state="expanded")

# --- Helper Functions ---

def get_subdirectories(path: Path) -> list[str]:
    """Returns a sorted list of subdirectory names within a given path."""
    if not path or not path.is_dir():
        return []
    try:
        return sorted([d.name for d in path.iterdir() if d.is_dir()])
    except FileNotFoundError:
        return []

@st.cache_data
def convert_df_to_csv(df):
    """Caches the conversion of a DataFrame to a CSV string for browser download."""
    return df.to_csv(index=False).encode('utf-8')

def select_folder():
    """Opens a native folder selection dialog using tkinter."""
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the main tkinter window
        root.attributes('-topmost', True)  # Make sure the dialog appears on top
        folder_selected = filedialog.askdirectory(master=root)
        root.destroy()
        return folder_selected
    except Exception:
        st.warning("Could not open folder dialog. This may happen in some environments. Please enter the path manually.")
        return None

def plot_time_coverage(stats: dict):
    """Creates a Gantt-style plot to visualize the time coverage of datasets."""
    if not stats: return None
    plot_data = []
    if 'cleaned' in stats and stats['cleaned']['start'] and stats['cleaned']['end']:
        plot_data.append(dict(Dataset="Full Dataset", Start=stats['cleaned']['start'], Finish=stats['cleaned']['end']))
    if 'train_val' in stats and stats['train_val']['start'] and stats['train_val']['end']:
        plot_data.append(dict(Dataset="Train / Validation", Start=stats['train_val']['start'], Finish=stats['train_val']['end']))
    if 'holdout' in stats and stats['holdout']['start'] and stats['holdout']['end']:
        plot_data.append(dict(Dataset="Holdout", Start=stats['holdout']['start'], Finish=stats['holdout']['end']))
    if not plot_data: return None

    df = pd.DataFrame(plot_data)
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Dataset", color="Dataset", title="<b>Data Time Coverage</b>")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(showlegend=False, plot_bgcolor="#f7f7f7")
    return fig

def get_df_stats(df, date_col="Point Name"):
    """Generates a stats dictionary needed for the time coverage plot."""
    if len(df) > 0:
        start = pd.to_datetime(df[date_col]).iloc[0]
        end = pd.to_datetime(df[date_col]).iloc[-1]
        return {"start": start, "end": end, "size": len(df)}
    return {"start": None, "end": None, "size": 0}

@st.cache_data
def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates descriptive statistics and null percentages for a DataFrame."""
    stats = df.describe(include='all').T
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / len(df)) * 100
    stats['null_%'] = null_percentages.round(2)
    return stats

# --- Sidebar Controls ---

with st.sidebar:
    st.header("1. Select Data Folder")

    if st.button("Browse for Data Folder", use_container_width=True):
        folder_path = select_folder()
        if folder_path:
            st.session_state.data_root_path = folder_path
            # Clear old selections when a new root folder is chosen
            for key in ['site_name', 'utility_name', 'sprint_name', 'model_name', 'selected_file', 'source_df', 'split_stats', 'show_stats', 'show_plot']:
                if key in st.session_state:
                    del st.session_state[key]
    
    data_root_path_str = st.session_state.get("data_root_path", "No folder selected")
    st.info(f"**Selected Folder:** `{data_root_path_str}`")
    
    path_is_valid = 'data_root_path' in st.session_state and Path(st.session_state.data_root_path).is_dir()

    # --- Hierarchical Drill-Down Selection ---
    if path_is_valid:
        st.divider()
        st.header("2. Project Configuration")
        root_path = Path(st.session_state.data_root_path)

        # Level 1: Site
        sites = get_subdirectories(root_path)
        if sites:
            st.selectbox("Select Site", sites, key="site_name", index=None, placeholder="Choose a site...")
        
        # Level 2: Utility
        if st.session_state.get("site_name"):
            site_path = root_path / st.session_state.site_name
            utilities = get_subdirectories(site_path)
            if utilities:
                st.selectbox("Select Utility", utilities, key="utility_name", index=None, placeholder="Choose a utility...")
        
        # Level 3: Sprint
        if st.session_state.get("utility_name"):
            utility_path = site_path / st.session_state.utility_name
            sprints = get_subdirectories(utility_path)
            if sprints:
                st.selectbox("Select Sprint", sprints, key="sprint_name", index=None, placeholder="Choose a sprint...")
        
        # Level 4: Model
        if st.session_state.get("sprint_name"):
            sprint_path = utility_path / st.session_state.sprint_name
            models = get_subdirectories(sprint_path)
            if models:
                st.selectbox("Select Model", models, key="model_name", index=None, placeholder="Choose a model...")

        # Level 5: Dataset File
        if st.session_state.get("model_name"):
            model_path = sprint_path / st.session_state.model_name
            dataset_path = model_path / "dataset"
            if dataset_path.is_dir():
                model_name = st.session_state.model_name
                files = list(dataset_path.glob(f"CLEANED-{model_name}-*-RAW.csv"))
                if files:
                    st.selectbox("Select Dataset File", [f.name for f in files], key="selected_file", index=None, placeholder="Choose a dataset file...")
            else:
                st.warning(f"A 'dataset' subfolder is missing in `{model_path}`.")
        
        preview_button_disabled = "selected_file" not in st.session_state or not st.session_state.selected_file
        preview_button = st.button("🔍 Load Data", use_container_width=True, disabled=preview_button_disabled)

    # Split configuration appears after data is loaded
    if 'source_df' in st.session_state:
        st.divider()
        st.header("3. Split Configuration")
        df = st.session_state.source_df
        split_type = st.radio("Split by:", ('Percentage', 'Specific Date'), horizontal=True, key="split_type")
        if split_type == 'Percentage':
            split_mark = st.slider("Holdout set size (%)", 1, 50, 20) / 100.0
        else:
            data_min_date = pd.to_datetime(df["Point Name"]).min().date()
            data_max_date = pd.to_datetime(df["Point Name"]).max().date()
            default_date = data_min_date + (data_max_date - data_min_date) * 0.8
            split_date = st.date_input("Select split date", value=default_date, min_value=data_min_date, max_value=data_max_date)
            split_mark = pd.to_datetime(split_date)
        run_button = st.button("🚀 Process and Split Data", use_container_width=True, type="primary")

# --- Main Page Display ---
st.title("📊 Data Splitting Module")

if 'preview_button' in locals() and preview_button:
    final_path = (Path(st.session_state.data_root_path) / 
                  st.session_state.site_name / st.session_state.utility_name / 
                  st.session_state.sprint_name / st.session_state.model_name / "dataset")
    raw_file = final_path / st.session_state.selected_file
    
    match = re.search(r'-(\d{8}-\d{8})-', str(raw_file.name))
    if match:
        st.session_state.inclusive_dates = match.group(1)
            
    with st.spinner(f"Loading file: {raw_file.name}"):
        df_raw = pd.read_csv(raw_file)
        st.session_state.source_df = df_raw.iloc[4:].reset_index(drop=True)
        st.session_state.df_header = df_raw.iloc[:4]
        st.session_state.dataset_path = final_path
        st.success(f"Successfully loaded file.")

if 'run_button' in locals() and run_button:
    if 'source_df' in st.session_state:
        with st.spinner("Splitting data..."):
            train_df, holdout_df, _, stats = split_holdout(
                cleaned_df=st.session_state.source_df, split_mark=split_mark, date_col="Point Name"
            )
            st.session_state.train_df = train_df
            st.session_state.holdout_df = holdout_df
            st.session_state.split_stats = stats
            st.success("Data split successfully!")

# --- Display Logic ---
if 'source_df' in st.session_state and 'split_stats' not in st.session_state:
    st.divider()
    st.subheader("Input Data Preview")
    st.dataframe(st.session_state.source_df.head(), use_container_width=True)

    st.divider()
    st.subheader("On-Demand Analysis")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Show Data Statistics", use_container_width=True):
            # Toggle the state
            st.session_state.show_stats = not st.session_state.get("show_stats", False)
    with col2:
        if st.button("📈 Show Time Coverage Plot", use_container_width=True):
            # Toggle the state
            st.session_state.show_plot = not st.session_state.get("show_plot", False)

    # Conditionally display the stats table
    if st.session_state.get("show_stats", False):
        st.subheader("Descriptive Statistics & Null Values")
        with st.spinner("Calculating statistics..."):
            stats_df = get_descriptive_stats(st.session_state.source_df)
            st.dataframe(stats_df)
    
    # Conditionally display the time coverage plot
    if st.session_state.get("show_plot", False):
        st.subheader("Initial Time Coverage")
        with st.spinner("Generating plot..."):
            preview_stats = {"cleaned": get_df_stats(st.session_state.source_df)}
            fig = plot_time_coverage(preview_stats)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not generate time coverage plot.")


if 'split_stats' in st.session_state:
    st.divider()
    st.subheader("Split Summary & Visualization")
    fig = plot_time_coverage(st.session_state.split_stats)
    if fig: st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    st.subheader("💾 Save Results to Disk")
    model_name = st.session_state.model_name
    inclusive_dates = st.session_state.get("inclusive_dates", "NODATES")
    train_val_out_fname = f"CLEANED-{model_name}-{inclusive_dates}-WITH-OUTLIER.csv"
    holdout_out_fname = f"{model_name}-{inclusive_dates}-HOLDOUT.csv"
    
    st.info(f"Clicking 'Save' will write files to `{st.session_state.dataset_path}`:")
    st.code(f"• {train_val_out_fname}\n• {holdout_out_fname}", language='bash')
    
    if st.button("💾 Save Split Files"):
        with st.spinner("Saving files..."):
            dataset_path = st.session_state.dataset_path
            dataset_path.mkdir(parents=True, exist_ok=True)
            train_val_out_path = dataset_path / train_val_out_fname
            holdout_out_path = dataset_path / holdout_out_fname
            df_train_to_save = pd.concat([st.session_state.df_header, st.session_state.train_df], ignore_index=True)
            df_holdout_to_save = pd.concat([st.session_state.df_header, st.session_state.holdout_df], ignore_index=True)
            df_train_to_save.to_csv(train_val_out_path, index=False)
            df_holdout_to_save.to_csv(holdout_out_path, index=False)
            st.success(f"Files saved successfully!")
            st.balloons()
            
    st.divider()
    st.subheader("Data Previews & Browser Download")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Train / Validation Set Preview")
        st.dataframe(st.session_state.train_df.head(), use_container_width=True)
        st.download_button("Download Train CSV", convert_df_to_csv(st.session_state.train_df), train_val_out_fname, use_container_width=True)
    with col2:
        st.write("Holdout Set Preview")
        st.dataframe(st.session_state.holdout_df.head(), use_container_width=True)
        st.download_button("Download Holdout CSV", convert_df_to_csv(st.session_state.holdout_df), holdout_out_fname, use_container_width=True)

