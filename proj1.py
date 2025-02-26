import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from cleaning import clean_empty
from regression import gradient_descent

def runProj1(fileName, outputFileName, alpha, iterations):
    df = pd.read_csv(fileName, na_values="########")
    df = clean_empty(df)
    
    X_vars = ['X1', 'X2', 'X3']
    y = df['Y'].values

    plt.figure(figsize=(12, 10))

    for i, var in enumerate(X_vars):
        X = df[[var]].values
        X = np.column_stack((np.ones(X.shape[0]), X))  # adding column of ones for the intercept

        theta_initial = np.zeros(X.shape[1])  
        theta_final, cost_history = gradient_descent(X, y, theta_initial, alpha, iterations)

        # regression model
        plt.subplot(3, 2, 2 * i + 1)
        plt.scatter(df[var], y, color='blue', label="Actual Data")
        plt.plot(df[var], np.dot(X, theta_final), color='red', label="Predicted")
        plt.xlabel(var)
        plt.ylabel("Y")
        plt.title(f"Regression Model for {var}")
        plt.legend()

        # loss over iterations
        plt.subplot(3, 2, 2 * i + 2)
        plt.plot(range(1, iterations + 1), cost_history, linestyle='-')
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title(f"Loss Over Iterations ({var})")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(outputFileName + ".png")
    
    print("final theta values:", theta_final)
    
    final_losses = {}

    for i, var in enumerate(X_vars):
        theta_initial = np.zeros(X.shape[1])  
        theta_final, cost_history = gradient_descent(X, y, theta_initial, alpha, iterations)
        
        final_losses[var] = cost_history[-1]  #store final loss as array
    
    best_var = min(final_losses, key=final_losses.get)
    print(f"The best explanatory variable is {best_var} with the lowest final loss: {final_losses[best_var]}")

    return
