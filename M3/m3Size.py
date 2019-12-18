import PIL
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import os
import os.path
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
import time
from M3 import m3Show


def makeSmaller(eye, show=True, scale=5, debug=True, minimum=50):
    h, w, c = eye.image.shape
    # print("originalShape", h,w,c)
    # mini
    downScaledDim = ((round(w/scale)), round(h/scale))

    inputImg = eye.wip.copy()
    inputImg = cv2.resize(inputImg, downScaledDim)
    # if inputImg.shape[1] > minimum
    eye.wip = inputImg
    m3Show.imshow(eye.wip, "makeSmaller")
    # eye.
    if debug:
        print("makeSmaller: Shape : ", inputImg.shape)
    return eye
