
import numpy as np
import pylint
# pyreverse -o png -p Pyreverse pylint/pyreverse/


def returnAttr(obj, attributeName):
    for attribute in obj.__dict__.items():
        if attribute[0] is attributeName:
            return attribute[1]  # return data for attribute


class Photo:
    originalImage = None
    loResImage = None
    path = None
    faces = []
    mask = None
    bigMask = None
    def __init__(self, originalImage="None", path="None"):
        self.originalImage = originalImage
        self.path = path
        self.mask = None

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


class Face:
    path = ""
    images = []
    eyes = []
    originalImage = None
    mask = None
    wip = None
    noFaceImg = None
    def __init__(self, eyes=[], noFaceImg=None):
        self.eyes = eyes
        self.noFaceImg = noFaceImg

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


class Eye:
    image = None
    cropRect = None
    landmarkPoints = None
    circle = None
    mask = None
    manyPolyMask = None
    polyMask = None
    noCircles = False
    houghOutline = None
    polyMaskedImage = None
    manyLandmarkPoints = []
    # **********************************************************************
    # wip = work in progress
    wip = None
    # **********************************************************************
    iris = None
    def __init__(self, im="None", cropRect="None", landmarkPoints="None"):
        self.image = im
        self.cropRect = cropRect
        self.landmarkPoints = landmarkPoints
        self.wip = np.copy(self.image)

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

#
# class Iris:
#     originalImage = None
#     circles = None
#     irisFromMask = None
#
#     def __init__(self, originalImage, irisFromMask):
#         originalImage = originalImage
#         irisFromMask = irisFromMask
