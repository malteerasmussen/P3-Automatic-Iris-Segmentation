import cv2, sys
import numpy as np
from matplotlib import pyplot as plt
sys.path.append("/M3")
from M3 import m3F as m3F
import os
from PIL import ImageFilter, ImageEnhance
import PIL


def medianFilter(inputImg, kernelSize, show):
    outputImg = cv2.medianBlur(inputImg, kernelSize)
    if (show):
        m3F.imshow(outputImg, "median blur")
    return outputImg


def blur(inputImg, kernelSize, show):
    # print("inputImg",inputImg)
    outputImg = inputImg.copy()
    outputImg = cv2.blur(inputImg, kernelSize, outputImg)
    if (show):
        m3F.imshow(outputImg, "median blur")
    return outputImg
