import sys
from PIL import ImageFilter
import cv2
sys.path.append("/M3")
from . import m3F


def gaussianBlur(inputImg, amount, show):
    # outputImg = inputImg.filter(ImageFilter.GaussianBlur(amount))
    outputImg = cv2.medianBlur(inputImg, amount)
    if (show):
        m3F.imshow(outputImg, "BLUR")
    return outputImg
