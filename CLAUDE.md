# EEG FORECASTING

This is a time series analysis project exploring the performance of various statistical models on forecast EEG data. The goal of this project is to write up publishable academic tutorials explaining the theoretical background of a model and demonstrating its use with a sample dataset. 

## TECH STACK
- Write all code in Python 3.11+. Use interpreter in `.venv` for all code execution.
- Algorithm background and tutorials applying to EEG are written in Jupyter notebooks.
- Leverages [MyST](https://mystmd.org/) folder structure to publish multiple tutorials as one book.
- Deep Learning methods should be implemented in `torch`.

## ARCHITECTURE
- All tutorials are in `/docs`.
- Utilty functions used across tutorials are in `/utils`.
- Github workflows for deploying tutorials are in `/.github`.
- Sample data provided in `dataset`.

## CODE CONVENTIONS
- Ensure all variables have a type declaration similar to Rust.
- Provide comments for each line or block of code. At least 1 comment per 5 lines of code minimum.
- Reusable functions across notebooks are stored in `/utils`.

## WRITE-UP CONVENTIONS
- Start with an opener about what the section will cover (see second cell of `/docs/DMD.ipynb`)
- Provide a **Background** section explaining the forecasting algorithm. Make sure to clearly explain the algorithm in natural language. Use equations to augment these explanations.
- Provide a section **Implementing** the algorithm. This is where the code will live.
- Include natural language explanations in between code cells to explain what is being done.
- Avoid massive code cells. They should only be 50-100 lines long max.
- Plot the results of forecasted values for a given `HORIZON`. Make sure to show the time series values that preceded the forecast and use a dotted vertical line to delineate the start of the horizon.
- End the notebook with a **Summary** section that distills the key points of the algorithm and how it performs at forecasting EEG data.
- Latex style equations are used for algorithm background info.

## TUTORIALS
1. `/docs/Intro.md`: Markdown file providing an introduction for the overall tutorial series. Motivates why forecasting EEG is an important area of research, and sets the stage for algorithms that will be explored.
2. `/docs/VARIMA.ipynb`: Forecast with **VARIMA** model.
3.  `/docs/DMD.ipynb`: Forecast with **Dynamic Mode Decomposition (DMD)** model.
4. `/docs/TCN.ipynb`: Forecast with **Temporal Convolutional Network (TCN)**.
5. `/docs/GRU.ipynb`: Forecast with **Gated Recurrent Network (GRU)**.
6. `/docs/Transformer.ipynb`: Forecast with **Transformer** model adapted for time series data (e.g., Informer, FEDFormer, etc.).
7. `/docs/S6.ipynb`: Forecast with **Selective State Space Model** (S6) (e.g, Mamba).

## RULES
- DO NOT push or merge with the git branch `main`. Only merge with the branch `dev` if the user permits.
- Create new git branches when modifying a notebook. Concisely specify what the modification is for in the git branch name.
- Ask for permission before installing new packages.
- Commit files to git before making edits that change more than 50 lines of code or text.
