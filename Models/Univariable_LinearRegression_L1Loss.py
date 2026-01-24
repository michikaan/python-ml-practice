import numpy as np
import matplotlib.pyplot as plt 

# Load data from text file
data = np.loadtxt("Models/q1data.txt", delimiter=",")
# Separate features and target variable
x=data[:,0]
y=data[:,1]
# Plot the data
plt.scatter(x,y)
plt.xlabel('City Population in 10,000s')
plt.ylabel('Montly Sales Revenue in $10,000s')
plt.title('City Population vs Sales Revenue')
plt.show()

#Compute the cost using L1 loss function
#function to compute L1 loss
def compute_cost_L1(x, y, theta0, theta1):
    m = len(y)
    total_cost = 0
    for i in range(m):
        prediction = theta0 + theta1 * x[i]
        total_cost += abs(prediction - y[i])
    return total_cost / m

#Gradient descent function to minimize L1 loss and also track cost history
def gradient_descent_L1(x, y, theta0, theta1, learning_rate, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        gradient0 = 0
        gradient1 = 0

        for i in range(m):
            prediction = theta0 + theta1 * x[i]
            error = prediction - y[i]

            if error > 0:
                gradient0 += 1
                gradient1 += x[i]
            elif error < 0:
                gradient0 -= 1
                gradient1 -= x[i]

        theta0 -= (learning_rate / m) * gradient0
        theta1 -= (learning_rate / m) * gradient1

        cost = compute_cost_L1(x, y, theta0, theta1)
        cost_history.append(cost)   

    return theta0, theta1, cost_history


# triying different learning rates
learning_rates = [0.003, 0.006, 0.014, 0.030]
iterations = 1000

plt.figure()

for lr in learning_rates:
    theta0, theta1, cost_history = gradient_descent_L1(
        x, y, 0, 0, lr, iterations
    )

    plt.plot(range(iterations), cost_history, label=f'lr={lr}')

plt.xlabel('Iterations')
plt.ylabel('L1 Loss (Cost)')
plt.title('L1 Loss vs Iterations for Different Learning Rates')
plt.legend()
plt.show()

#final parameters after gradient descent with a chosen learning rate=0.014
final_theta0, final_theta1, _ = gradient_descent_L1(x, y, 0, 0, 0.014, iterations)
print(f"Final parameters after gradient descent: theta0 = {final_theta0}, theta1 = {final_theta1}")
# Plotting the final regression line
plt.figure()
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, final_theta0 + final_theta1 * x, color='red', label='Regression Line')
plt.xlabel('City Population')
plt.ylabel('Montly Sales Revenue')
plt.title('Linear Regression Fit using L1 Loss')
plt.legend()
plt.show()

# Making predictions for given data points
x1 = 2
x2 = 6

yhat1 = final_theta0 + final_theta1 * x1
yhat2 = final_theta0 + final_theta1 * x2

print(f"Prediction for x = {x1}: y = {yhat1}")
print(f"Prediction for x = {x2}: y = {yhat2}")

-
# L1 Cost Surface (theta0, theta1)

theta0_vals = np.linspace(-10, 10, 120)
theta1_vals = np.linspace(-1, 4, 120)

T0, T1 = np.meshgrid(theta0_vals, theta1_vals)
Jgrid = np.zeros(T0.shape)

for i in range(T0.shape[0]):
    for j in range(T0.shape[1]):
        Jgrid[i, j] = compute_cost_L1(
            x, y, T0[i, j], T1[i, j]
        )


# 3D Surface Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(T0, T1, Jgrid, cmap='viridis', edgecolor='none')
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('L1 Cost')
ax.set_title('L1 Cost Surface')

plt.show()


# Contour Plot with Optimum

plt.figure()
contours = plt.contour(T0, T1, Jgrid, levels=40)
plt.clabel(contours, inline=True, fontsize=8)

plt.plot(final_theta0, final_theta1, 'rx', markersize=10, linewidth=2,
         label='Optimum')

plt.xlabel('theta0')
plt.ylabel('theta1')
plt.title('L1 Cost Contours (Optimum Marked)')
plt.legend()
plt.grid(True)
plt.show()
