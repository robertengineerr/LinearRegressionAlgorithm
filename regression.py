import pandas as pd
import numpy as np

def gradient_descent(x, y, theta, learning_rate, iterations):
    cost_history=[]

    # steps
    for i in range(iterations):
        h_theta = np.dot(x, theta)  # dot product of x and theta
        error = h_theta - y  # distance between h and y for error correction
        
        cost = (1/(2*len(y))) * np.sum((h_theta - y)**2)
        cost_history.append(cost)
        #get new theta
        theta = theta - (learning_rate/len(y)) * np.dot(x.T, error)
    
    #storing cost history to a .txt file for debugging
    cost_df = pd.DataFrame({"Iteration": np.arange(1, iterations + 1), "Cost": cost_history})
    cost_df.to_csv('cost_history.csv', index=False)
    
    return theta
