# Linear Regression from Scratch

**Author:** Robert Thomas  
**Course:** ECGR-4105-001 – Machine Learning  
**Date:** February 25, 2025

This project implements a linear regression algorithm in Python using gradient descent from scratch—no scikit-learn, no prebuilt ML libraries. It was developed as part of a class assignment to reinforce a deeper understanding of how linear models work at the algorithmic level.

---

## What It Does

- Implements linear regression using custom gradient descent
- Tests both single-variable and multivariable regression
- Cleans real-world CSV input data (handles missing values)
- Visualizes cost convergence and learning rate impact using Matplotlib
- Predicts outcomes for new data points using trained models

---

## Files

- `linear_regression_main.py` – Command-line interface and execution entry
- `regression.py` – Core linear regression and prediction functions
- `proj1.py` – Part 1: separate models for each input variable (X1, X2, X3)
- `proj2.py` – Part 2: combined model using all input variables
- `cleaning.py` – Replaces missing values in the dataset
- `help.txt` – CLI help documentation

---

## How to Run

### Part 1 – Train separate models for each input (X1, X2, X3)

```bash
python3 linear_regression_main.py -p 1 -f D3.csv
