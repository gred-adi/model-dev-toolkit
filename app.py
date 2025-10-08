# app.py
import streamlit as st

st.set_page_config(
    page_title="Data Analysis Workbench",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Data Analysis Workbench")

st.markdown("""
Welcome to the interactive data analysis app! This tool is designed to walk you through a complete data science workflow, from cleaning to reporting.

**This app allows you to:**
-   Upload and clean time-series data from PRISM CSV files.
-   Split your data into training and holdout sets.
-   Detect outliers using various statistical and machine learning methods.
-   Analyze correlations between different features.
-   Generate and download reports.

### How to Use
1.  Navigate to the **1_Data_Cleaning** page using the sidebar.
2.  Upload your CSV file.
3.  Proceed through the pages in order to perform your analysis.
""")

st.sidebar.success("Select a page above to begin.")