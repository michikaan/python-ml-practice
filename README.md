# Linear Regression – From Scratch (Python)

This repository contains **from-scratch implementations of linear regression** in Python.
The purpose of this project is to understand the mathematical foundations of linear
regression and optimization methods without relying on machine learning libraries.

---

## 1. Basic Linear Regression (From Scratch)

This example implements a simple linear regression model using the equation:

y = a·x + b

### Description
- The slope (`a`) and intercept (`b`) are defined manually.
- Model predictions are computed explicitly using loops.
- Raw data and model predictions are visualized using matplotlib.

### Files
- `linear_regression.py`  
  Implements a basic linear regression model and visualization.

---

## 2. Linear Regression with L1 Loss (MAE)

This example implements **univariable linear regression using L1 loss (Mean Absolute Error)**
and **batch gradient descent with subgradient optimization**.

### Description
- The hypothesis function is defined as:

  h(x) = θ₀ + θ₁x

- The cost function is the L1 loss (MAE):

  J(θ) = (1/m) ∑ |h(xᵢ) − yᵢ|

- Since L1 loss is not differentiable at zero, **subgradient methods** are used.
- Batch gradient descent is implemented from scratch.
- Different learning rates are tested and compared.
- Final regression line is plotted over the data.
- Predictions are made for unseen data points.
- The L1 cost surface and contour plots are visualized, and the optimum point is marked.

### Files
- `Models/Univariable_LinearRegression_L1Loss.py`  
  Full implementation of L1 linear regression with gradient descent and visualization.

- `Models/q1data.txt`  
  Dataset used for training and evaluation.

---

## Libraries Used

- NumPy
- Matplotlib

---

## Outputs

- Scatter plots of raw data
- Regression line overlays
- Cost vs. iteration plots for different learning rates
- 3D surface plot of the L1 cost function
- Contour plot with optimal parameters marked

---

## Purpose

This repository is part of a personal machine learning practice project.
It focuses on building strong intuition for regression, cost functions,
and optimization techniques before moving to high-level libraries.


