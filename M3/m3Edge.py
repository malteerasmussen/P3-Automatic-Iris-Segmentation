import cv2
from M3 import m3Show


def sobel(inputImg, show=True):
    out = cv2.Sobel(inputImg, inputImg.depth(), 0, 1, 3)
    if show:
        m3Show.imshow(out, "show")


def canny(inputImg, show=True, thresh1=100, thresh2=100):
    out = cv2.Canny(inputImg, thresh1, thresh2)
    if show:
        m3Show.imshow(out, "canny")
    return out

def maskOutEdges(eye, show=True):
    pass
