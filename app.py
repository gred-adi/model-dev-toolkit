import streamlit as st

st.set_page_config(
    page_title="Model Development Toolkit",
    page_icon="🛠️",
    layout="wide"
)

st.title("🛠️ Model Development Toolkit")

st.markdown("""
Welcome to the interactive Model Development Toolkit! This application is a multi-page utility designed to guide you through a comprehensive data processing and analysis workflow, from initial data splitting to final accuracy calculation.

Each page in the sidebar represents a distinct step in the model development lifecycle. Please proceed through them in order.
""")

st.divider()

st.header("Workflow Guide")

st.markdown("""
**1. `Data Splitting`**
-   **Purpose:** To perform the initial split of your raw dataset into a **training/validation set** and a **holdout set**.
-   **Action:** Select your project folder and the raw `CLEANED...-RAW.csv` file. The app will partition the data based on your chosen date or percentage, preparing it for the cleaning process.

**2. `Outlier Detection`**
-   **Purpose:** To interactively identify and remove outliers from the training/validation set created in the previous step.
-   **Action:** Load the `CLEANED...-WITH-OUTLIER.csv` file. Analyze each feature against the operational state, visualize potential outliers, and remove them iteratively. Save the final cleaned dataset.

**3. `Split Training & Validation`**
-   **Purpose:** To perform a stratified split on the cleaned dataset, separating it into final **training** and **validation** sets.
-   **Action:** The app requires both the `WITH-OUTLIER` and `WITHOUT-OUTLIER` datasets, along with configuration files. It uses a continuous stratification method to ensure both sets have similar distributions.

**4. `Calculate Accuracy`**
-   **Purpose:** To process the model's output (`.dat` file) and calculate the final accuracy and relative deviation for each metric.
-   **Action:** Select the project folder to locate the model's `.dat` file and the `fault_detection.csv` config. The app will parse the data and generate a final accuracy report.
""")

st.sidebar.success("Select a page above to begin.")