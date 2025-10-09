import streamlit as st
import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import re

st.set_page_config(page_title="Calculate Accuracy", layout="wide")

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

def extract_numeric(value):
    """Extracts the numeric part from a string like 'text, 1.23'."""
    try:
        return float(str(value).split(', ')[1])
    except (IndexError, ValueError):
        return float('nan')

# --- Main App ---
st.title("📈 Calculate Accuracy")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("1. Select Project Folder")
    if st.button("Browse for Data Folder", use_container_width=True):
        folder_path = select_folder()
        if folder_path:
            st.session_state.data_root_path_acc = folder_path
            # Clear state on new folder selection
            for key in ['site_name', 'utility_name', 'sprint_name', 'model_name', 'accuracy_df']:
                if key in st.session_state:
                    del st.session_state[key]

    st.info(f"**Folder:** `{st.session_state.get('data_root_path_acc', 'Not Selected')}`")
    path_is_valid = 'data_root_path_acc' in st.session_state and Path(st.session_state.data_root_path_acc).is_dir()

    if path_is_valid:
        st.divider()
        st.header("2. Project Configuration")
        root_path = Path(st.session_state.data_root_path_acc)

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
            st.header("3. Calculate Accuracy")
            model_name = st.session_state.model_name
            model_path = sprint_path / model_name
            
            # Define paths to required files
            deviation_path = model_path / "relative_deviation"
            config_path = model_path / "config"
            dat_file = deviation_path / f"{model_name}.dat"
            fault_file = config_path / "fault_detection.csv"

            st.write("Required Input Files:")
            st.markdown(f"- `{model_name}.dat`: {'✅' if dat_file.exists() else '❌'}")
            st.markdown(f"- `fault_detection.csv`: {'✅' if fault_file.exists() else '❌'}")
            
            load_ready = all([dat_file.exists(), fault_file.exists()])
            if st.button("🚀 Load & Calculate", use_container_width=True, disabled=not load_ready, type="primary"):
                st.session_state.accuracy_df = None # Clear previous results
                with st.spinner("Loading files and calculating accuracy..."):
                    # Load data
                    df_data = pd.read_csv(dat_file, encoding="UTF-16", delimiter='\\t')
                    fault_df = pd.read_csv(fault_file)

                    # Clean column names
                    col_map = {col: re.split(r'\(Virtual|\(Arkanghel|\(', col)[0].strip() for col in df_data.columns}
                    df_data.rename(columns=col_map, inplace=True)
                    
                    # Get metric list
                    metrics = fault_df.columns.tolist()
                    if 'Name' in metrics: metrics.remove('Name')
                    if 'Minimum OMR' in metrics: metrics.remove('Minimum OMR')
                    
                    # Filter data to only include relevant metrics
                    available_metrics = [m for m in metrics if m in df_data.columns]
                    df_filtered = df_data[available_metrics]
                    
                    # Calculate accuracy
                    df_numeric = df_filtered.apply(lambda col: col.apply(extract_numeric))
                    results = [{
                        'Metrics': col,
                        'Average Relative Deviation (%)': round(abs(df_numeric[col].mean()) * 100, 2),
                        'Accuracy (%)': round((1 - abs(df_numeric[col].mean())) * 100, 2),
                    } for col in df_numeric.columns]
                    
                    st.session_state.accuracy_df = pd.DataFrame(results)
                    st.success("Accuracy calculation complete.")

# --- Main Page Display ---
if 'accuracy_df' not in st.session_state or st.session_state.accuracy_df is None:
    st.info("Please select a project and run the calculation using the sidebar.")
else:
    df_scores = st.session_state.accuracy_df
    
    st.subheader("Model Accuracy Report")
    
    avg_accuracy = df_scores['Accuracy (%)'].mean()
    st.metric("Overall Average Accuracy", f"{avg_accuracy:.2f}%")
    
    st.dataframe(df_scores, use_container_width=True)

    # Save Results
    st.divider()
    st.subheader("💾 Export Report")
    
    # FIX: Use .get() to safely access session state variables and provide defaults.
    model_name = st.session_state.get("model_name", "MODEL")
    site_name = st.session_state.get("site_name", "SITE")
    utility_name = st.session_state.get("utility_name", "UTILITY")
    sprint_name = st.session_state.get("sprint_name", "SPRINT")
    data_root_path_acc = st.session_state.get("data_root_path_acc", ".")

    model_path = Path(data_root_path_acc) / site_name / utility_name / sprint_name / model_name
    save_path = model_path / "relative_deviation"
    output_fname = f"{model_name}_Accuracy.csv"

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Report to Disk", use_container_width=True):
            with st.spinner("Saving..."):
                save_path.mkdir(exist_ok=True)
                df_scores.to_csv(save_path / output_fname, index=False)
                st.success("Report saved successfully!")
                st.balloons()
        st.info(f"Report will be saved as: `{save_path / output_fname}`")
    with col2:
        st.download_button(
            label="Download Report as CSV",
            data=df_scores.to_csv(index=False).encode('utf-8'),
            file_name=output_fname,
            mime='text/csv',
            use_container_width=True
        )

