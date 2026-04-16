# Didaskalos Streamlit App

This folder contains a production-oriented Streamlit app for the Didaskalos workflow.

## What it does

- Lists available XML treebanks
- Lets you select one or more files
- Builds a combined token dataframe
- Computes a frequency-based syllabus table
- Exports CSV files
- Generates textbook markdown from your lesson modules

## Project layout

- app.py: Streamlit UI
- didaskalos_pipeline.py: reusable data and export functions
- requirements.txt: Python dependencies
- .streamlit/config.toml: Streamlit runtime/theme config

## Run locally

1. Open a terminal at the Didaskalos repository root.
2. Install dependencies:

   pip install -r didaskalos_streamlit_app/requirements.txt

3. Run the app:

   streamlit run didaskalos_streamlit_app/app.py

## Default folders expected

- Treebanks: treebanks/perseus
- Lesson modules: lessons-no-decl

You can change both paths in the app sidebar.
