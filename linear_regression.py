import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. Data
# -------------------------------------------------
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.2, 2.8, 3.6, 4.5, 5.1])

print(f"x = {x}")
print(f"y = {y}")

# Number of training samples
n = len(x)
print(f"Number of training data is: {n}")

# -------------------------------------------------
# 2. Plot raw data
# -------------------------------------------------
plt.figure()
plt.scatter(x, y, marker='x', color='red')
plt.title("Random Data")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# -------------------------------------------------
# 3. Linear model parameters (y = ax + b)
# -------------------------------------------------
a = 0.7   # slope
b = 1.5   # intercept

def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model y = wx + b

    Args:
        x (ndarray): input data (m samples)
        w (float): slope
        b (float): intercept

    Returns:
        ndarray: model predictions
    """
    m = x.shape[0]
    y_pred = np.zeros(m)

    for i in range(m):
        y_pred[i] = w * x[i] + b

    return y_pred

# Model prediction
y_pred = compute_model_output(x, a, b)

# -------------------------------------------------
# 4. Plot data + model
# -------------------------------------------------
plt.figure()

plt.scatter(x, y, marker='x', color='red', label='Actual Data')
plt.plot(x, y_pred, color='blue', label='Model Prediction')

plt.title("Linear Regression Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.show()
