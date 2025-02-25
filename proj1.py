import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from cleaning import clean_empty
from regression import gradient_descent

def runProj1(fileName, outputFileName, alpha, iterations):
    df = pd.read_csv(fileName, na_values="########")

    df = clean_empty(df)
    
    X = df[['X1', 'X2', 'X3']].values
    y = df['Y'].values
    
    X = np.column_stack((np.ones(X.shape[0]), X))  # adding a column of ones for the intercept. required for the linear regression formula
    
    #parameters
    theta_initial = np.zeros(X.shape[1])  # initial values for theta
    
    theta_final, cost_history = gradient_descent(X, y, theta_initial, alpha, iterations)
    
    cost_df = pd.read_csv("cost_history.csv")
    plt.plot(cost_df['Iteration'], cost_df['Cost'])  # plotting the cost over iterations
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title(f'Cost History during Gradient Descent\nα = {alpha}, Iterations = {iterations}')
    plt.savefig(outputFileName+".png")
    
    print("final theta values:", theta_final)
    return