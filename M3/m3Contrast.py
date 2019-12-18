import cv2, sys
import numpy as np
from matplotlib import pyplot as plt
sys.path.append("/M3")
from M3 import m3F as m3F
import os
from PIL import ImageFilter, ImageEnhance
import PIL
from M3 import m3Show


def contrast(inputImg, show):
    inputImg = PIL.Image.fromarray(inputImg)
    outputImg = ImageEnhance.Contrast(inputImg).enhance(1.4)
    if (show):
        m3Show.imshow(outputImg, "Contrast")
    outputImg = np.asarray(outputImg)
    return outputImg
