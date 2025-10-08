import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import datetime
import os
import tkinter as tk
from tkinter import filedialog

# Assuming your utility functions are in the utils directory
from utils.data_utils import split_holdout

st.set_page_config(page_title="Data Splitting", layout="wide", initial_sidebar_state="expanded")

# --- Helper Functions ---

@st.cache_data
def convert_df_to_csv(df):
    """Caches the conversion of a DataFrame to a CSV string for browser download."""
    return df.to_csv(index=False).encode('utf-8')

def select_folder():
    """Opens a native folder selection dialog."""
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    # Make sure the dialog appears on top of other windows
    root.attributes('-topmost', True)
    # Open the dialog to ask for a directory
    folder_selected = filedialog.askdirectory(master=root)
    root.destroy()
    return folder_selected

def plot_time_coverage(stats: dict):
    # ... (function is unchanged) ...
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
    # ... (function is unchanged) ...
    if len(df) > 0:
        start = pd.to_datetime(df[date_col]).iloc[0]
        end = pd.to_datetime(df[date_col]).iloc[-1]
        return {"start": start, "end": end, "size": len(df)}
    return {"start": None, "end": None, "size": 0}


# --- Sidebar Controls ---

with st.sidebar:
    st.header("1. Select Data Folder")

    # NEW: Use a button to trigger the tkinter folder dialog
    if st.button("Browse for Data Folder", use_container_width=True):
        folder_path = select_folder()
        if folder_path:  # Check if the user selected a folder
            st.session_state.data_root_path = folder_path

    # Display the selected path using a disabled text_input
    data_root_path_str = st.text_input(
        "Selected Path",
        value=st.session_state.get("data_root_path", "No folder selected"),
        disabled=True
    )
    
    path_is_valid = 'data_root_path' in st.session_state and Path(st.session_state.data_root_path).is_dir()

    st.divider()
    st.header("2. Project Configuration")

    site_name = st.text_input("Site Name", value="TVI")
    utility_name = st.text_input("Utility Name", value="BOP")
    sprint_name = st.text_input("Sprint Name", value="Sprint_1")
    model_name = st.text_input("Model Name", value="AP-TVI-U1-BFP_A_MOTOR")
    inclusive_dates = st.text_input("Inclusive Dates (YYYYMMDD-YYYYMMDD)", value="20240601-20250801")
    
    # Disable button if no valid folder is selected
    preview_button = st.button("🔍 Load & Preview Data", use_container_width=True, disabled=not path_is_valid)

    # The Split Configuration section appears after data is loaded
    if 'source_df' in st.session_state:
        st.divider()
        st.header("3. Split Configuration")
        # ... (Rest of the split configuration logic is unchanged) ...
        try:
            start_date_str, end_date_str = inclusive_dates.split('-')
            date_range_valid = True
        except (ValueError, IndexError):
            date_range_valid = False
            st.warning("Invalid format for inclusive dates.")

        split_type = st.radio("Split by:", ('Percentage', 'Specific Date'), horizontal=True, key="split_type")
        
        if split_type == 'Percentage':
            split_mark = st.slider("Holdout set size (%)", 1, 50, 20) / 100.0
        else: # By Specific Date
            if date_range_valid:
                df = st.session_state.source_df
                data_min_date = pd.to_datetime(df["Point Name"]).min().date()
                data_max_date = pd.to_datetime(df["Point Name"]).max().date()
                default_date = data_min_date + (data_max_date - data_min_date) * 0.8
                split_date = st.date_input("Select split date", value=default_date, min_value=data_min_date, max_value=data_max_date)
                split_mark = pd.to_datetime(split_date)
            else:
                split_mark = None
                st.error("Cannot set split date without valid inclusive dates.")

        run_button = st.button("🚀 Process and Split Data", use_container_width=True, type="primary", disabled=(not date_range_valid))


# --- Main Page Display ---
# ... (The rest of the Main Page Display logic is unchanged) ...
st.title("📊 Data Splitting Module")

if preview_button:
    data_root_path = Path(st.session_state.data_root_path)
    dataset_path = data_root_path / site_name / utility_name / sprint_name / model_name / "dataset"
    raw_fname = f"CLEANED-{model_name}-{inclusive_dates}-RAW.csv"
    raw_file = dataset_path / raw_fname
    
    for key in ['train_df', 'holdout_df', 'split_stats', 'source_df']:
        if key in st.session_state: del st.session_state[key]
            
    with st.spinner(f"Searching for file: {raw_file}"):
        if not raw_file.exists():
            st.error(f"File Not Found! Ensure the following file exists: `{raw_file}`")
        else:
            df_raw = pd.read_csv(raw_file)
            st.session_state.source_df = df_raw.iloc[4:].reset_index(drop=True)
            st.session_state.df_header = df_raw.iloc[:4]
            st.session_state.dataset_path = dataset_path
            st.success(f"Located and loaded file: `{raw_file}`")

if 'run_button' in locals() and run_button:
    if 'source_df' in st.session_state:
        with st.spinner("Splitting data..."):
            train_df, holdout_df, _, stats = split_holdout(
                cleaned_df=st.session_state.source_df,
                split_mark=split_mark,
                date_col="Point Name"
            )
            st.session_state.train_df = train_df
            st.session_state.holdout_df = holdout_df
            st.session_state.split_stats = stats
            st.success("Data split successfully!")
    else:
        st.warning("Please load data first using the 'Load & Preview Data' button.")

if 'source_df' in st.session_state and 'split_stats' not in st.session_state:
    st.divider()
    st.subheader("Input Data Preview")
    st.dataframe(st.session_state.source_df.head(), use_container_width=True)
    st.subheader("Initial Time Coverage")
    preview_stats = {"cleaned": get_df_stats(st.session_state.source_df)}
    fig = plot_time_coverage(preview_stats)
    if fig: st.plotly_chart(fig, use_container_width=True)

if 'split_stats' in st.session_state:
    st.divider()
    st.subheader("Split Summary & Visualization")
    fig = plot_time_coverage(st.session_state.split_stats)
    if fig: st.plotly_chart(fig, use_container_width=True)
    st.divider()
    st.subheader("💾 Save Results to Disk")
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