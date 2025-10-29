import streamlit as st
import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import os
import shutil
import logging # Use streamlit's logging or python's basic logging

# --- Import Core QA Functions ---
# !!! Ensure these utils are copied from model-qa-tools repo into the ./utils/ folder !!!
try:
    from utils.qa_reporting import generate_qa_report
    from utils.qa_ks_comparison import compare_data_distributions
    from utils.qa_plotting import generate_report_plots
    # Add any other necessary imports from the qa utils
except ImportError as e:
    st.error(f"Missing QA utility files in './utils/'. Please copy them from the model-qa-tools repository. Error: {e}")
    # Stop the script if utils are missing
    st.stop() 

# Import general utils
from utils.data_utils import select_folder, get_subdirectories

st.set_page_config(page_title="Model QA", layout="wide")

# --- Helper Functions ---

def get_model_file_paths(base_path: Path, sprint: str, model: str) -> dict:
    """
    Locates required files for a specific model based on expected structure.
    Returns a dictionary of paths, with None for missing files.
    """
    model_path = base_path / sprint / model
    dataset_path = model_path / "dataset"
    split_path = model_path / "data_splitting"
    config_path = model_path / "config"
    perf_dir = model_path / "performance_assessment_report"
    fpr_dir = perf_dir / "FPR"
    ks_dir = perf_dir / "KS"
    report_dir = perf_dir / "report_document"

    paths = {
        "model_path": model_path,
        "dataset_path": dataset_path,
        "split_path": split_path,
        "config_path": config_path,
        "perf_dir": perf_dir,
        "fpr_path": fpr_dir,
        "ks_path": ks_dir,
        "report_path": report_dir,
        "raw_data": None,
        "holdout_data": None,
        "val_without_outlier": None,
        "val_with_outlier": None, # May be needed by some functions
        "holdout_omr": None,      # Optional OMR files
        "val_without_outlier_omr": None,
        "val_with_outlier_omr": None,
        "points_file": config_path / "project_points.csv",
        "constraints_file": None # This will be selected by user
    }

    # Find specific data files
    if dataset_path.is_dir():
        raw_files = list(dataset_path.glob(f"{model}-*-RAW.csv"))
        holdout_files = list(dataset_path.glob(f"{model}-*-HOLDOUT.csv"))
        if raw_files: paths["raw_data"] = raw_files[0]
        if holdout_files: paths["holdout_data"] = holdout_files[0]
        # Optional OMR files in dataset
        hold_omr_files = list(dataset_path.glob(f"OMR-{model}-*-HOLDOUT.csv"))
        if hold_omr_files: paths["holdout_omr"] = hold_omr_files[0]


    if split_path.is_dir():
        val_wo_files = list(split_path.glob(f"VALIDATION-{model}-*-WITHOUT-OUTLIER.csv"))
        val_w_files = list(split_path.glob(f"VALIDATION-{model}-*-WITH-OUTLIER.csv"))
        if val_wo_files: paths["val_without_outlier"] = val_wo_files[0]
        if val_w_files: paths["val_with_outlier"] = val_w_files[0]
        # Optional OMR files in split path
        val_wo_omr_files = list(split_path.glob(f"OMR-VALIDATION-{model}-*-WITHOUT-OUTLIER.csv"))
        val_w_omr_files = list(split_path.glob(f"OMR-VALIDATION-{model}-*-WITH-OUTLIER.csv"))
        if val_wo_omr_files: paths["val_without_outlier_omr"] = val_wo_omr_files[0]
        if val_w_omr_files: paths["val_with_outlier_omr"] = val_w_omr_files[0]

    # Check existence of essential config file
    paths["points_file_exists"] = paths["points_file"].exists()

    return paths

# --- Main App ---
st.title("✅ Model Quality Assurance Report Generator")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("1. Select Project Folder")
    if st.button("Browse for Root Data Folder", use_container_width=True):
        folder_path = select_folder()
        if folder_path:
            st.session_state.data_root_path_qa_report = folder_path
            # Clear state
            for key in ['site_name', 'utility_name', 'sprint_name', 'model_name', 'qa_report_files_found', 'qa_report_generated']:
                if key in st.session_state: del st.session_state[key]

    st.info(f"**Folder:** `{st.session_state.get('data_root_path_qa_report', 'Not Selected')}`")
    path_is_valid = 'data_root_path_qa_report' in st.session_state and Path(st.session_state.data_root_path_qa_report).is_dir()

    if path_is_valid:
        st.divider()
        st.header("2. Project Configuration")
        root_path = Path(st.session_state.data_root_path_qa_report)

        # Hierarchical selection
        sites = get_subdirectories(root_path)
        if sites: st.selectbox("Site", sites, key="site_name", index=None)
        if st.session_state.get("site_name"):
            site_path = root_path / st.session_state.site_name
            utilities = get_subdirectories(site_path)
            if utilities: st.selectbox("Utility", utilities, key="utility_name", index=None)
        if st.session_state.get("utility_name"):
            utility_path = site_path / st.session_state.utility_name
            sprints = get_subdirectories(utility_path)
            if sprints: st.selectbox("Sprint", sprints, key="sprint_name", index=None)
        if st.session_state.get("sprint_name"):
            sprint_path = utility_path / st.session_state.sprint_name
            models = get_subdirectories(sprint_path)
            if models: st.selectbox("Model", models, key="model_name", index=None)

        # File check and parameter input
        if st.session_state.get("model_name"):
            st.divider()
            st.header("3. QA Parameters & Files")
            
            # --- Locate Files ---
            paths = get_model_file_paths(root_path, st.session_state.sprint_name, st.session_state.model_name)
            st.session_state.qa_paths = paths # Store paths for later use

            st.write("**Required Data Files:**")
            st.markdown(f"- Raw Data: {'✅' if paths['raw_data'] else '❌'}")
            st.markdown(f"- Holdout Data: {'✅' if paths['holdout_data'] else '❌'}")
            st.markdown(f"- Validation (Clean): {'✅' if paths['val_without_outlier'] else '❌'}")
            st.write("**Required Config File:**")
            st.markdown(f"- `project_points.csv`: {'✅' if paths['points_file_exists'] else '❌'}")
            
            # Optional OMR files (just indicate presence)
            st.write("**Optional OMR Files:**")
            st.markdown(f"- Holdout OMR: {'✅' if paths['holdout_omr'] else '