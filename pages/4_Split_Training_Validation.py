import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from verstack.stratified_continuous_split import scsplit

st.set_page_config(page_title="Split Training & Validation", layout="wide")

# --- Helper Functions ---

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

def get_subdirectories(path: Path) -> list[str]:
    """Returns a sorted list of subdirectory names within a given path."""
    if not path or not path.is_dir():
        return []
    return sorted([d.name for d in path.iterdir() if d.is_dir()])

def read_prism_csv(path):
    """Reads a PRISM CSV, separating the 4-row header from the data."""
    df = pd.read_csv(path)
    header = df.iloc[:4]
    data = df.iloc[4:].reset_index(drop=True)
    data['Point Name'] = pd.to_datetime(data['Point Name'], errors='coerce')
    # Convert all other columns to numeric, coercing errors
    for col in data.columns:
        if col != 'Point Name':
            data[col] = pd.to_numeric(data[col], errors='coerce')
    return data, header

# --- Main App ---

st.title("🔀 Split Training & Validation Sets")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("1. Select Project Folder")
    if st.button("Browse for Data Folder", use_container_width=True):
        folder_path = select_folder()
        if folder_path:
            st.session_state.data_root_path_split = folder_path
            # Clear state on new folder selection
            for key in ['site_name', 'utility_name', 'sprint_name', 'model_name', 'split_data_loaded']:
                if key in st.session_state:
                    del st.session_state[key]
    
    st.info(f"**Folder:** `{st.session_state.get('data_root_path_split', 'Not Selected')}`")
    path_is_valid = 'data_root_path_split' in st.session_state and Path(st.session_state.data_root_path_split).is_dir()

    if path_is_valid:
        st.divider()
        st.header("2. Project Configuration")
        root_path = Path(st.session_state.data_root_path_split)

        # Hierarchical drill-down selection
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
        
        # File check and load button
        if st.session_state.get("model_name"):
            st.divider()
            st.header("3. Load Data")
            model_path = sprint_path / st.session_state.model_name
            dataset_path = model_path / "dataset"
            config_path = model_path / "config"
            
            # Find files based on patterns
            model_name = st.session_state.model_name
            with_outlier_files = list(dataset_path.glob(f"CLEANED-{model_name}-*-WITH-OUTLIER.csv"))
            without_outlier_files = list(dataset_path.glob(f"CLEANED-{model_name}-*-WITHOUT-OUTLIER.csv"))
            
            st.selectbox("Select file WITH outliers", [f.name for f in with_outlier_files], key="file_with_outlier")
            st.selectbox("Select file WITHOUT outliers", [f.name for f in without_outlier_files], key="file_without_outlier")
            
            # Check for config files
            points_file = config_path / "project_points.csv"
            fault_file = config_path / "fault_detection.csv"

            st.write("Required Config Files:")
            st.markdown(f"- `project_points.csv`: {'✅' if points_file.exists() else '❌'}")
            st.markdown(f"- `fault_detection.csv`: {'✅' if fault_file.exists() else '❌'}")
            
            load_ready = all([with_outlier_files, without_outlier_files, points_file.exists(), fault_file.exists()])
            if st.button("💾 Load & Prepare Data", use_container_width=True, disabled=not load_ready):
                st.session_state.split_data_loaded = False
                # Load all files
                with st.spinner("Loading and processing files..."):
                    st.session_state.df_with, st.session_state.header_with = read_prism_csv(dataset_path / st.session_state.file_with_outlier)
                    st.session_state.df_without, st.session_state.header_without = read_prism_csv(dataset_path / st.session_state.file_without_outlier)
                    points_df = pd.read_csv(points_file)
                    fault_df = pd.read_csv(fault_file)

                    # Process column mapping and selection from notebook
                    mapping = dict(zip(points_df["Name"], points_df["Metric"]))
                    st.session_state.df_without.rename(columns=mapping, inplace=True)
                    
                    fault_cols = fault_df.columns.tolist()
                    fault_cols[0] = 'Point Name'
                    if 'Minimum OMR' in fault_cols: fault_cols.remove('Minimum OMR')
                    st.session_state.fault_cols = fault_cols
                    
                    st.session_state.df_selected = st.session_state.df_without[fault_cols]

                    # Find operational state for stratification
                    op_state = points_df[points_df['Constrain'] == True]['Metric'].values[0]
                    st.session_state.op_state = op_state

                    st.session_state.split_data_loaded = True
                    st.success("Data loaded and prepared.")

    if st.session_state.get("split_data_loaded"):
        st.divider()
        st.header("4. Split Configuration")
        st.info(f"Stratifying by: **{st.session_state.op_state}**")
        test_size = st.slider("Validation Set Size", 0.1, 0.5, 0.2, 0.05)
        
        if st.button("🚀 Run Split", use_container_width=True, type="primary"):
            with st.spinner("Performing stratified split..."):
                
                # FIX: Ensure the stratification column is purely numeric before splitting
                df_for_split = st.session_state.df_selected.copy()
                op_state_col = st.session_state.op_state
                
                # Coerce to numeric, turning non-numeric strings into NaN
                df_for_split[op_state_col] = pd.to_numeric(df_for_split[op_state_col], errors='coerce')
                
                # Drop rows where the op_state is NaN to prevent errors in scsplit
                original_rows = len(df_for_split)
                df_for_split.dropna(subset=[op_state_col], inplace=True)
                cleaned_rows = len(df_for_split)

                if original_rows > cleaned_rows:
                    st.warning(f"Removed {original_rows - cleaned_rows} rows with non-numeric data in the stratification column ('{op_state_col}') before splitting.")

                train, validate = scsplit(df_for_split, stratify=df_for_split[op_state_col], test_size=test_size, continuous=True)
                
                st.session_state.train_df = st.session_state.df_without[st.session_state.df_without['Point Name'].isin(train['Point Name'])]
                st.session_state.val_df_without = st.session_state.df_without[st.session_state.df_without['Point Name'].isin(validate['Point Name'])]
                st.session_state.val_df_with = st.session_state.df_with[~st.session_state.df_with['Point Name'].isin(train['Point Name'])]
                st.session_state.split_run_complete = True
                
# --- Main Page Display ---
if not st.session_state.get("split_data_loaded"):
    st.info("Please select a project and load data using the sidebar.")
elif not st.session_state.get("split_run_complete"):
    st.subheader("Data Ready for Splitting")
    st.write("Columns selected for modeling:")
    st.dataframe(st.session_state.fault_cols)
    st.subheader("Preview of Data (Without Outliers)")
    st.dataframe(st.session_state.df_selected.head())
else:
    st.subheader("Split Results")
    st.metric("Training Set Shape", str(st.session_state.train_df.shape))
    st.metric("Validation Set (Without Outliers) Shape", str(st.session_state.val_df_without.shape))
    st.metric("Validation Set (With Outliers) Shape", str(st.session_state.val_df_with.shape))

    # Boxplot Visualization
    st.subheader("Train vs. Validation Distribution")
    with st.spinner("Generating boxplot..."):
        # FIX: Explicitly convert data to numeric before melting to avoid TypeErrors
        train_numeric = st.session_state.train_df.drop(columns=['Point Name']).apply(pd.to_numeric, errors='coerce')
        val_numeric = st.session_state.val_df_without.drop(columns=['Point Name']).apply(pd.to_numeric, errors='coerce')

        df_plot = pd.concat({
            'TRAIN': train_numeric.melt(), 
            'VALIDATION': val_numeric.melt()
        }, names=['data', 'old_index']).reset_index(level=0).reset_index(drop=True)
        
        g = sns.catplot(data=df_plot, kind='box', x='data', y='value', col='variable', col_wrap=6, height=3, aspect=1.25, sharey=False, palette={'TRAIN': 'blue', 'VALIDATION': 'orange'})
        g.set(xlabel='', xticks=[])
        g.set_titles('{col_name}', size=8.5)
        st.pyplot(g)

    # Save Results
    st.divider()
    st.subheader("💾 Save Split Datasets")
    model_path = Path(st.session_state.data_root_path_split) / st.session_state.site_name / st.session_state.utility_name / st.session_state.sprint_name / st.session_state.model_name
    save_path = model_path / "data_splitting"
    
    model_name = st.session_state.model_name
    fname_without = st.session_state.file_without_outlier
    fname_with = st.session_state.file_with_outlier
    
    st.info(f"Files will be saved to: `{save_path}`")
    if st.button("Save Files to Disk", use_container_width=True):
        with st.spinner("Saving..."):
            save_path.mkdir(exist_ok=True)
            
            # Prepare and save each file
            train_export = pd.concat([st.session_state.header_without, st.session_state.train_df])
            train_export.to_csv(save_path / fname_without.replace('CLEANED', 'TRAINING'), index=False)
            
            val_without_export = pd.concat([st.session_state.header_without, st.session_state.val_df_without])
            val_without_export.to_csv(save_path / fname_without.replace('CLEANED', 'VALIDATION'), index=False)

            val_with_export = pd.concat([st.session_state.header_with, st.session_state.val_df_with])
            val_with_export.to_csv(save_path / fname_with.replace('CLEANED', 'VALIDATION'), index=False)
            
            st.success("All 3 files saved successfully!")
            st.balloons()

