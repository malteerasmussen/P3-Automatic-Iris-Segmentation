import cv2
import numpy as np
from matplotlib import pyplot as plt

def Histogram(img):
    
    #load image and split the image into the color channels
    img = img
    b, g, r = cv2.split(img)
    
    
    #Creates a histogram of the different channels
    plt.hist(b.ravel(), 256, [0, 256])
    plt.hist(g.ravel(), 256, [0, 256])
    plt.hist(r.ravel(), 256, [0, 256])
    BGR_Histogram =plt.show()
    return (BGR_Histogram)
 




   

