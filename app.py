import streamlit as st

st.set_page_config(
    page_title="Model Development Toolkit",
    page_icon="🛠️",
    layout="wide"
)

st.title("🛠️ Model Development Toolkit")

st.markdown("""
Welcome to the interactive Model Development Toolkit! This application is a multi-page utility designed to guide you through a comprehensive data processing and analysis workflow, from initial data splitting to final accuracy calculation and model QA.

Each page in the sidebar represents a distinct step in the model development lifecycle. Please proceed through them in order.
""")

st.divider()

st.header("Workflow Guide")

st.markdown("""
**1. `Data Splitting`**
-   **Purpose:** Perform the initial split of your raw dataset into a **training/validation set** and a **holdout set**.
-   **Action:** Select project folder and raw `CLEANED...-RAW.csv`. Partitions data by date/percentage. Outputs: `...-WITH-OUTLIER.csv` and `...-HOLDOUT.csv`.

**2. `Outlier Detection`**
-   **Purpose:** Interactively identify and remove outliers from the training/validation set.
-   **Action:** Load `...-WITH-OUTLIER.csv`. Analyze features, visualize outliers, confirm removal. Output: `...-WITHOUT-OUTLIER.csv`.

**3. `Split Training & Validation`**
-   **Purpose:** Perform a stratified split on the cleaned dataset into final **training** and **validation** sets.
-   **Action:** Requires `WITH-OUTLIER`, `WITHOUT-OUTLIER`, and config files. Uses stratification. Outputs: `TRAINING...csv`, `VALIDATION...WITH-OUTLIER.csv`, `VALIDATION...WITHOUT-OUTLIER.csv`.

**4. `Calculate Accuracy`**
-   **Purpose:** Process model output (`.dat` file) and calculate accuracy/relative deviation.
-   **Action:** Select project, finds `.dat` and `fault_detection.csv`. Parses data, generates accuracy report. Output: `..._Accuracy.csv`.

**5. `Model QA`** (New)
-   **Purpose:** Perform Quality Assurance by comparing model predictions against actual values from the validation set.
-   **Action:** Select project, finds validation set and prediction files. Calculates metrics (MAE, MSE, R², etc.) and generates comparison plots. Output: QA Report/Plots.
""")

st.sidebar.success("Select a page above to begin.")