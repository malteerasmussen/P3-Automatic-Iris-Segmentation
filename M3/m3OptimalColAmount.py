import numpy as np
import matplotlib.pyplot as plt
import csv

def findoptimalcolours(csvFilesPath):
    
    for file in csvFilesPath:
        print(file)
    
    colorAmount = []
    y_means = []
        
    for file in csvFilesPath:
        colorAmount.append(file.colors)
        y = 0
        
        for eye in file.eyes:
            y += (FP+FN)/(TP+1)
            
        y_means.append(y/file.eyes.len)
        
    
    
    plt.