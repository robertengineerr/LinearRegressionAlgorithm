import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from cleaning import clean_empty
from regression import gradient_descent, predict

def runProj2(fileName, outputFileName, alpha, iterations):
    learning_rates = []
    for i in range(1, 50):
        learning_rates.append(i*alpha/50)
    
    df = pd.read_csv(fileName, na_values="########")

    df = clean_empty(df)
    
    X = df[['X1', 'X2', 'X3']].values
    y = df['Y'].values
    
    X = np.column_stack((np.ones(X.shape[0]), X))  # adding a column of ones for the intercept. required for the linear regression formula
    
    theta_initial = np.zeros(X.shape[1])  # initial values for theta
    
    results = {}
    cost_values=[]
    cost_histories = {}

    for lr in learning_rates:
        theta_final, cost_history = gradient_descent(X, y, theta_initial, lr, iterations)
        results[lr] = theta_final
        cost_values.append(cost_history[-1])
        cost_histories[lr] = cost_history
    
    best_lr = learning_rates[np.argmin(cost_values)]
    best_cost_history = cost_histories[best_lr]
    
    # Plot Learning Rate vs. Final Cost
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(learning_rates, cost_values, marker='o', linestyle='-')
    plt.xlabel('Learning Rate')
    plt.ylabel('Final Cost')
    plt.title('Effect of Learning Rate on Final Cost')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, iterations + 1), best_cost_history, marker='.', linestyle='-')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title(f'Cost Over Iterations (Best LR={best_lr:.4f})')
    plt.grid(True)
    
    plt.savefig(outputFileName + ".png")
    
    print("final theta values:", theta_final)

    X_new_values = [
        [1, 1, 1],
        [2, 0, 4],
        [3, 2, 1],
    ]
    
    for X_new in X_new_values:
        y_pred = predict(theta_final, X_new)
        print(f"Predicted y for input {X_new}: {y_pred}")

    return