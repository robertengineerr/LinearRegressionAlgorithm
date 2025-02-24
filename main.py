import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from cleaning import clean_empty
from regression import gradient_descent

df = pd.read_csv("D3.csv", na_values="########")

df = clean_empty(df)

X = df[['X1', 'X2', 'X3']].values
y = df['Y'].values

X = np.column_stack((np.ones(X.shape[0]), X))  # adding a column of ones for the intercept. required for the linear regression formula

#parameters
theta_initial = np.zeros(X.shape[1])  # Starting values for theta
alpha = 0.01  # learning rate
iterations = 1000

theta_final = gradient_descent(X, y, theta_initial, alpha, iterations)

cost_df = pd.read_csv('cost_history.csv')
plt.plot(cost_df['Iteration'], cost_df['Cost'])  # Plot the cost over iterations
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost History during Gradient Descent')
plt.show()

print("final theta values:", theta_final)