import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from cleaning import clean_empty
from regression import gradient_descent
from proj1 import runProj1

fileName="D3.csv"
alpha=0.05
iterations = 1000
outputFileName = 'cost_history_plot'

whichProj=1

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
        elif sys.argv[i]=="-p":
            if i + 1 < len(sys.argv):
                whichProj = int(sys.argv[i+1])
        elif sys.argv[i] == "--help":
            print("-a   Alpha value (learning rate)")
            print("-i   Number of iterations")
            print("-f   Input csv file name")
            print("-o   Output plot file name")

if whichProj==1:
    runProj1(fileName, outputFileName, alpha, iterations)
elif whichProj==2:
    print("Not done yet")
    #runProj2()