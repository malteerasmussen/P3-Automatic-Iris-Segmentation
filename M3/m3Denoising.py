import cv2, sys
import numpy as np
from matplotlib import pyplot as plt
sys.path.append("/M3")
from M3 import m3F as m3F
import os
from PIL import ImageFilter, ImageEnhance
import PIL


def denoise(inputImg, show):
    outputImg = cv2.fastNlMeansDenoisingColored(inputImg,7,21,10,10)
    if (show):
        m3F.imshow(outputImg, "Denoised")
    return outputImg