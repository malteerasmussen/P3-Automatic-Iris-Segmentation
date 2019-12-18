import cv2, sys
import numpy as np
from matplotlib import pyplot as plt
sys.path.append("/M3")
from M3 import m3F as m3F
import os
from PIL import ImageFilter, ImageEnhance
import PIL
import math


def segmentEyelid(eye, show):
    iris = eye.wip
    m3F.imshow(iris, "iris")
    gray = cv2.cvtColor(iris, cv2.COLOR_BGR2GRAY)
    c = cv2.Canny(iris, 100,iris.shape[0]*1.5)
    m3F.imshow(c, "canny")
    print("AARG")

    sobel_vertical = cv2.Sobel(c, -1, 0, 1, ksize=3)
    m3F.imshow(sobel_vertical, "Sobel")

    print("sobel",sobel_vertical)

    indices = np.where(sobel_vertical != [0])
    coordinates = zip(indices[1], indices[0])

    #print("indices",indices)

    print("shape", iris.shape[0])



    for i in eye.circle[0, :]:
        #print("x,y",x,y)
        for x,y in coordinates:
            #if (y < i[1]):
                #print("What is i?",i[0],i[1])
            if math.sqrt(((x-i[0])**2)+((y-i[1])**2)) < i[2]:
                cv2.circle(iris,(x,y),1,(0,255,0))

        m3F.imshow(iris, "Found edges")


    return iris
