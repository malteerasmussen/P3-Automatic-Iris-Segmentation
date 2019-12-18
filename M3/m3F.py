# __name__ = "__main__"

import PIL
from matplotlib import pyplot as plt
import numpy as np
import os
import os.path
import cv2
import json


# http://ozzmaker.com/add-colour-to-text-in-python/


def star():
    print("***********************************************************************")


def printRed(stringarr):
    #prstr = ""
    # for el in stringarr:
    #    prstr + str(el)
    prstr = "\033[1;41;39m" + stringarr + "\]"
    print(prstr)


def printGreen(stringarr):
    prstr = "\033[1;42;39m" + stringarr + "\]"
    print(prstr)


def printBlue(stringarr):
    prstr = "\033[1;44;39m" + stringarr + "\]"
    print(prstr)


def evalSize(imagePath, w, h):
    nd = np.ndarray(1)
    if (type(imagePath) == type(nd)):
        im = imagePath
        im = typeSwap(im)
    elif(type(imagePath) == type(str)):
        im = PIL.Image.open(imagePath)
        im = typeSwap(im)

    # print("W & H", im.width, im.height)
    if (im.width > w and im.height > h):
        # was larger,
        return True
    else:
        # too small
        return False


# https://pythontic.com/image-processing/pillow/histogram
def getRed(redVal):
    return '#%02x%02x%02x' % (redVal, 0, 0)


def getGreen(greenVal):
    return '#%02x%02x%02x' % (0, greenVal, 0)


def getBlue(blueVal):
    return '#%02x%02x%02x' % (0, 0, blueVal)


def gHist(imagePath):
    # print(type(imagePath))
    pimg = PIL.Image.fromarray(imagePath)
    hist = pimg.histogram()
    # plt.hist(hist,bins=256)
    plt.figure(0)
    for i in range(0, 256):
        plt.bar(i, hist[i], alpha=0.3)
    plt.show()


# swaps input between numpy ndArray (as OPENCV USES) and PIL image

def typeSwap(inputIm):
    # printBlue("typeswap:")
    nd = np.ndarray(1)
    im = PIL.Image.Image()
    #print(type(inputIm), type(nd))
    if (type(inputIm) == type(nd)):
        inputIm = PIL.Image.fromarray(inputIm)
        return inputIm
    elif(type(inputIm) == type(im)):
        inputIm = np.asarray(inputIm)
        return inputIm


def filecount(dir_name):
    return len([f for f in os.listdir(dir_name)if os.path.isfile(os.path.join(dir_name, f))])


# shows image in notebook

def imshow(inputImg, title):
    plt.title(title)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB))
    plt.show()


def getRed(inputImg, show):
    (channel_b, channel_g, channel_r) = cv2.split(inputImg)
    channel_r = cv2.cvtColor(channel_r, cv2.COLOR_GRAY2BGR)
    if (show):
        imshow(channel_r,"red")
    return channel_r

# **********************************************************************
# **********************************************************************


#  Return an attribute of an object dynamically.
#  This allows us to use an attribute as a parameter for a function.
#  i.e. accessessing an eye's attribute, like wip, iris.
#  this is needed as you won't able to just pass the parameter (x) like eye.x
def rattr(obj, attributeName):
    for attribute in obj.__dict__.items():
        if attribute[0] is attributeName:
            return attribute[1]  # return data for attribute


def storeAttr(photo, eyeAttr, attrName):
    for face in photo.faces:
        for eye in face.eyes:
            setattr(eye, attrName, rattr(eye.eyeAttr, attrName))
    return photo


# **********************************************************************
# **********************************************************************

def funcArrToStr(settings):
    dict = []
    for el in settings:
        for function in el:
            print(function.__name__)
            e = {function.__name__: el[function]}
            dict.append(e)
    return dict
