import streamlit as st
import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import re

# Assuming your utility functions are in the utils directory
# For this self-contained example, they are assumed to exist.
# from utils.outlier_utils import detect_outliers
# from utils.viz_utils import plot_outlier_detection_plotly

# --- Placeholder for utility functions ---
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def detect_outliers(
    df, op_state_col, feature, method='isoforest', contamination=0.01, n_neighbors=20, iqr_factor=1.5, percentile_cut=0.01
):
    X = df[[op_state_col, feature]].copy().apply(pd.to_numeric, errors='coerce').dropna()
    idx = X.index
    if method == 'isoforest':
        model = IsolationForest(contamination=contamination, random_state=0)
        outlier_flag = model.fit_predict(X)
    elif method == 'lof':
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        outlier_flag = lof.fit_predict(X)
    # ... other methods if needed ...
    else:
        raise ValueError("Unsupported method.")
    flags = pd.Series(1, index=df.index) # Default to inlier
    flags.loc[idx[outlier_flag == -1]] = -1 # Mark outliers
    return flags

def plot_outlier_detection_plotly(df, date_col, op_state_col, feature, outlier_flag):
    X = df[[date_col, op_state_col, feature]].dropna().copy()
    X['outlier'] = outlier_flag.loc[X.index]
    inliers = X[X['outlier'] == 1]
    outliers = X[X['outlier'] == -1]
    
    # FIX: Added the 'specs' argument to enable the secondary y-axis on the first subplot.
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=False, 
        vertical_spacing=0.1, 
        subplot_titles=("Time Series Overview", "Feature vs. Operational State"),
        specs=[[{"secondary_y": True}], [{}]]
    )
    
    fig.add_trace(go.Scatter(x=X[date_col], y=X[op_state_col], name=op_state_col, line=dict(color='green'), mode="lines"), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=X[date_col], y=X[feature], name=feature, line=dict(color='blue'), mode="lines"), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=outliers[date_col], y=outliers[feature], name="Outlier", mode="markers", marker=dict(color='red', size=8)), row=1, col=1, secondary_y=True)
    
    fig.add_trace(go.Scatter(x=inliers[op_state_col], y=inliers[feature], mode='markers', marker=dict(color='gray', size=5, opacity=0.6), name="Inlier"), row=2, col=1)
    fig.add_trace(go.Scatter(x=outliers[op_state_col], y=outliers[feature], mode='markers', marker=dict(color='red', size=7, line=dict(color='black', width=1)), name="Outlier"), row=2, col=1)
    
    fig.update_layout(height=800, plot_bgcolor="#f7f7f7", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig
# --- End of placeholder functions ---

def get_subdirectories(path: Path) -> list[str]:
    """Returns a sorted list of subdirectory names within a given path."""
    if not path or not path.is_dir():
        return []
    return sorted([d.name for d in path.iterdir() if d.is_dir()])

def select_folder():
    """Opens a native folder selection dialog."""
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder_selected = filedialog.askdirectory(master=root)
        root.destroy()
        return folder_selected
    except Exception:
        return None

st.set_page_config(page_title="Outlier Detection", layout="wide")

# --- Initialize Session State ---
if 'outlier_runs' not in st.session_state:
    st.session_state.outlier_runs = []

# --- Sidebar Controls ---
with st.sidebar:
    st.header("1. Select Data Folder")

    if st.button("Browse for Data Folder", use_container_width=True):
        folder_path = select_folder()
        if folder_path:
            st.session_state.data_root_path_outlier = folder_path
            # Clear old selections when a new folder is chosen
            for key in ['site_name', 'utility_name', 'sprint_name', 'model_name', 'selected_file_outlier', 'train_df', 'outlier_runs']:
                if key in st.session_state:
                    del st.session_state[key]

    data_root_path_str = st.session_state.get("data_root_path_outlier", "No folder selected")
    st.info(f"**Selected Folder:** `{data_root_path_str}`")
    path_is_valid = 'data_root_path_outlier' in st.session_state and Path(st.session_state.data_root_path_outlier).is_dir()

    if path_is_valid:
        st.divider()
        st.header("2. Project Configuration")
        root_path = Path(st.session_state.data_root_path_outlier)

        # Hierarchical drill-down selection
        sites = get_subdirectories(root_path)
        if sites: st.selectbox("Select Site", sites, key="site_name", index=None, placeholder="Choose a site...")
        if st.session_state.get("site_name"):
            site_path = root_path / st.session_state.site_name
            utilities = get_subdirectories(site_path)
            if utilities: st.selectbox("Select Utility", utilities, key="utility_name", index=None, placeholder="Choose a utility...")
        if st.session_state.get("utility_name"):
            utility_path = site_path / st.session_state.utility_name
            sprints = get_subdirectories(utility_path)
            if sprints: st.selectbox("Select Sprint", sprints, key="sprint_name", index=None, placeholder="Choose a sprint...")
        if st.session_state.get("sprint_name"):
            sprint_path = utility_path / st.session_state.sprint_name
            models = get_subdirectories(sprint_path)
            if models: st.selectbox("Select Model", models, key="model_name", index=None, placeholder="Choose a model...")
        
        # File selection for the specific input required by this page
        if st.session_state.get("model_name"):
            model_path = sprint_path / st.session_state.model_name / "dataset"
            if model_path.is_dir():
                model_name = st.session_state.model_name
                files = list(model_path.glob(f"CLEANED-{model_name}-*-WITH-OUTLIER.csv"))
                if files:
                    st.selectbox("Select Input File", [f.name for f in files], key="selected_file_outlier", index=None, placeholder="Choose a dataset...")
            else:
                st.warning(f"A 'dataset' subfolder is missing in `{model_path.parent}`.")
        
        load_button_disabled = "selected_file_outlier" not in st.session_state or not st.session_state.selected_file_outlier
        load_data_button = st.button("💾 Load Data for Outlier Detection", use_container_width=True, disabled=load_button_disabled)

    if 'train_df' in st.session_state:
        st.divider()
        st.header("3. Outlier Detection Setup")
        df = st.session_state.train_df
        # Use all column headers for selection lists
        all_cols = df.columns.tolist()
        
        # Exclude the date column from operational state options
        op_state_options = [col for col in all_cols if col != "Point Name"]

        st.subheader("Operational State")
        op_state_col = st.selectbox("Select Operational State Column", op_state_options)

        st.subheader("Feature Analysis")
        # Exclude both date and the selected op_state column from feature options
        available_features = [col for col in all_cols if col not in ["Point Name", op_state_col]]
        feature_to_analyze = st.selectbox("Select Feature to Analyze", available_features)
        method = st.selectbox("Detection Method", ['isoforest', 'lof'])
        contamination = st.slider("Contamination", 0.0001, 0.1, 0.0005, 0.0001, format="%.4f")
        run_detection_button = st.button("🔍 Run Detection", use_container_width=True, type="primary")

# --- Main Page Display ---
st.title("🔬 Outlier Detection Module")

if 'load_data_button' in locals() and load_data_button:
    full_path = (Path(st.session_state.data_root_path_outlier) / 
                 st.session_state.site_name / st.session_state.utility_name / 
                 st.session_state.sprint_name / st.session_state.model_name / "dataset")
    input_file = full_path / st.session_state.selected_file_outlier
    
    with st.spinner(f"Loading {input_file.name}..."):
        df_raw = pd.read_csv(input_file)
        st.session_state.train_df = df_raw.iloc[4:].reset_index(drop=True) # Use main df key
        st.session_state.df_header = df_raw.iloc[:4]
        st.session_state.dataset_path = full_path
        
        # Extract inclusive_dates from filename
        match = re.search(r'-(\d{8}-\d{8})-', input_file.name)
        if match:
            st.session_state.inclusive_dates = match.group(1)
            
        st.success(f"Successfully loaded data. You may now configure the outlier detection in the sidebar.")

if 'train_df' not in st.session_state:
    st.info("Please select and load a dataset using the sidebar to begin.")
else:
    df = st.session_state.train_df # Define df here for access in the logic below

    if 'run_detection_button' in locals() and run_detection_button:
        # The operational state filtering logic has been removed.
        with st.spinner(f"Running {method} on '{feature_to_analyze}'..."):
            flags = detect_outliers(df, op_state_col, feature_to_analyze, method=method, contamination=contamination)
            st.session_state.current_outliers = flags
            st.session_state.last_run_feature = feature_to_analyze
            num_outliers = (flags == -1).sum()
            st.success(f"Detection complete. Found **{num_outliers}** potential outliers for '{feature_to_analyze}'.")

    if 'current_outliers' in st.session_state:
        st.divider()
        feature = st.session_state.last_run_feature
        st.subheader(f"Analysis for: `{feature}`")
        fig = plot_outlier_detection_plotly(df, "Point Name", op_state_col, feature, st.session_state.current_outliers)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Remove Detected Outliers")
        num_outliers = (st.session_state.current_outliers == -1).sum()
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"Original training set size: `{df.shape[0]}` rows.")
            st.write(f"Number of outliers to be removed: `{num_outliers}` rows.")
            st.write(f"New training set size: `{df.shape[0] - num_outliers}` rows.")
        with col2:
            if st.button("✅ Confirm Removal", use_container_width=True, disabled=(num_outliers == 0)):
                inlier_indices = st.session_state.current_outliers[st.session_state.current_outliers != -1].index
                st.session_state.train_df = df.loc[inlier_indices].copy()
                run_info = {"feature": feature, "removed_count": num_outliers, "new_shape": st.session_state.train_df.shape}
                st.session_state.outlier_runs.append(run_info)
                del st.session_state.current_outliers
                st.success(f"Removed {num_outliers} outliers. You can now select another feature to analyze.")
                st.rerun()

    if st.session_state.outlier_runs:
        st.divider()
        st.subheader("Summary of Removed Outliers")
        summary_df = pd.DataFrame(st.session_state.outlier_runs)
        st.dataframe(summary_df, use_container_width=True)

        st.subheader("💾 Save Cleaned Data")
        if 'dataset_path' in st.session_state:
            model_name = st.session_state.get('model_name', 'MODEL')
            inclusive_dates = st.session_state.get('inclusive_dates', 'NODATES')
            cleaned_fname = f"CLEANED-{model_name}-{inclusive_dates}-WITHOUT-OUTLIER.csv"
            st.info(f"This will save the final cleaned data to `{st.session_state.dataset_path / cleaned_fname}`.")
            if st.button("💾 Save to File", use_container_width=True):
                with st.spinner("Saving file..."):
                    df_final = pd.concat([st.session_state.df_header, st.session_state.train_df], ignore_index=True)
                    output_path = st.session_state.dataset_path / cleaned_fname
                    df_final.to_csv(output_path, index=False)
                    st.success(f"File saved successfully!")
                    st.balloons()