import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D

def rgbToHSV(inputImg):
    
    hsv_pic = cv2.cvtColor(inputImg, cv2.COLOR_RGB2HSV)
    
    #Splits into color channels and creates a 3D plot 
    h,s,v = cv2.split(hsv_pic)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    
    #Pixels colors and normalize 
    pixel_colors = inputImg.reshape((np.shape(inputImg)[0]*np.shape(inputImg)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    
    #plots into H, S and V
    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")

    result = plt.show()
    return (inputImg)
 




   

