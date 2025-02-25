import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from cleaning import clean_empty
from regression import gradient_descent

fileName="D3.csv"
alpha=0.05
iterations = 1000
outputFileName = 'cost_history_plot'

if len(sys.argv) > 1:
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == "-a":
            if i + 1 < len(sys.argv):
                alpha=float(sys.argv[i+1])
        elif sys.argv[i] == "-i":
            if i + 1 < len(sys.argv):
                iterations = int(sys.argv[i+1])
        elif sys.argv[i] == "-f":
            if i + 1 < len(sys.argv):
                fileName=sys.argv[i+1]
        elif sys.argv[i]=="-o":
            if i + 1 < len(sys.argv):
                outputFileName = sys.argv[i+1]
        elif sys.argv[i] == "--help":
            print("-a   Alpha value (learning rate)")
            print("-i   Number of iterations")
            print("-f   Input csv file name")
            print("-o   Output plot file name")

df = pd.read_csv(fileName, na_values="########")

df = clean_empty(df)

X = df[['X1', 'X2', 'X3']].values
y = df['Y'].values

X = np.column_stack((np.ones(X.shape[0]), X))  # adding a column of ones for the intercept. required for the linear regression formula

#parameters
theta_initial = np.zeros(X.shape[1])  # initial values for theta

theta_final = gradient_descent(X, y, theta_initial, alpha, iterations)

cost_df = pd.read_csv("cost_history.csv")
plt.plot(cost_df['Iteration'], cost_df['Cost'])  # plotting the cost over iterations
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title(f'Cost History during Gradient Descent\nα = {alpha}, Iterations = {iterations}')
plt.savefig(outputFileName+".png")

print("final theta values:", theta_final)