

from M3 import m3Show
from M3 import m3Class
from M3 import m3F as m3F
import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt
sys.path.append("/M3")



def makeCircularOutline(eye, circles, show=False, dest=None):
    # for face in photo.faces:
        # for eye in face.eyes:
    maskImg = eye.image.copy()
    if not isinstance(eye.circle, type(None)):
        # firstCircle = eye.circle[0]
        for i in circles[0, :]:
            print("i", i)
            cv2.circle(maskImg, (i[0], i[1]), i[2], (255, 255, 255), 2)
            # eye.mask = maskImg
            # setattr(eye, dest, maskImg)
            if (show):
                m3Show.imshow(maskImg, "mask")
    return maskImg













def fullImgEyeOutline(photo, show):
    fullMask = photo.originalImage.copy()
    x, y, channels = fullMask.shape
    for face in photo.faces:
        for eye in face.eyes:
            if not isinstance(eye.mask, type(None)):
                coor = eye.cropRect
                # fullMask[coor[0]:x, coor[1]:y] = eye.mask

                for i in eye.circle[0, :]:
                    print("i", i)
                    cv2.circle(fullMask, (coor[0]+i[0], coor[1]+i[1]), i[2], (0, 255, 0), 2)

                    print("coor", coor)
                    #fullMask[coor[1]:coor[1]+eye.mask.shape[0], coor[0]:coor[0]+eye.mask.shape[1]] = eye.mask
    photo.mask = fullMask
    if (show):
        m3Show.imshow(photo.originalImage, "full original")
        m3Show.imshow(photo.mask, "full mask")
    return photo














def makeFullMask(photoArray, eyeAttr, show=False):
    for photo in photoArray:
        fullMask = photo.originalImage.copy()
        fullMask.fill(0)

        # img1 = cv.imread('messi5.jpg')
        # img2 = cv.imread('opencv-logo-white.png')
        # I want to put logo on top-left corner, So I create a ROI
        x, y, channels = fullMask.shape
        for face in photo.faces:
            for eye in face.eyes:
                attr = getattr(eye, eyeAttr)
                if not isinstance(attr, type(None)):
                    coor = eye.cropRect
                    # fullMask[coor[0]:x, coor[1]:y] = eye.mask
                    # print("coor", coor)
                    fullMask[coor[1]:coor[1]+attr.shape[0], coor[0]:coor[0]+attr.shape[1]] = attr
                    photo.fullMask = fullMask
                    if (show):
                        pass
                        # m3Show.imshow(inputImg.orginalImage, "full original")
                        # m3Show.imshow(fullMask, "full mask")
    return photoArray







# def makePolyMask(photo, show=True):
#     # polyMask =  np.zeros_like(photo.originalImage)
#     for face in photo.faces:
#         for eye in face.eyes:
#             polyMask =  np.zeros_like(eye.image)
#             print("EYE COOR", eye.landmarkPoints)
#             polyMask = cv2.fillPoly(polyMask, np.int_([eye.landmarkPoints]), (255, 255, 255))
#             # polymask = cv2.fillConvexPoly(polyMask, np.int_([eye.landmarkPoints]), (255, 255, 255),1000)
#             # polymask = cv2.drawContours(polyMask, np.int_([eye.landmarkPoints]),-1, (255, 255, 255), 2)
#             # polymask = cv2.approxPolyDP(np.int_([eye.landmarkPoints]),4, True)
#             # xs, ys = [], []
#             # for x, y in eye.eye.landmarkPoints:
#             #     xs.append(x)
#             #     ys.append(y)
#             # polymask = cv2.poly
#             # print(polymask)
#             if show:
#                 m3Show.imshow(polyMask, "POLYMASK")
#             # epm = m3Class.Eye(np.asarray(pil_image.crop(left)), left, lEyeCoor)
#             # epm = polyMask.crop(eye.cropRect)
#             # eye.polyMask = m3F.typeSwap(m3F.typeSwap(polyMask).crop(eye.cropRect))
#             eye.polyMask = polyMask
#             if show:
#                 m3Show.imshow(eye.polyMask, "poly")
#
#     return photo


# p


def makeManyPolyMask(photo, show=True):
    # **********************************************************************
    # polyMask =  np.zeros_like(photo.originalImage)
    # **********************************************************************
    for face in photo.faces:
        for eye in face.eyes:
            polyMask =  np.zeros_like(eye.image)
            print("manyLandmarkPoints", eye.manyLandmarkPoints)
            polyMask = cv2.fillPoly(polyMask, np.int_([eye.manyLandmarkPoints]), (255, 255, 255))
            # m3Show.imshow(polyMask, "POLYMASK")
            # epm = m3Class.Eye(np.asarray(pil_image.crop(left)), left, lEyeCoor)
            # epm = polyMask.crop(eye.cropRect)
            # eye.manyPolyMask = m3F.typeSwap(m3F.typeSwap(polyMask).crop(eye.cropRect))
            eye.manyPolyMask = polyMask
            # if show:
            m3Show.imshow(eye.manyPolyMask, "manypoly")

    return photo




def applyPolyMask(eye, show=True):
    polymask = None
    eye.wip = cv2.bitwise_and(eye.wip, eye.polyMask)
    # eye.wip = cv2.bitwise_and(eye.wip, eye.manyPolyMask)
    if show:
        m3Show.imshow(eye.wip, "masked")
    return eye

# for attr in eye.__dict__.items():
#     if attr


# import attr






def returnAttr(obj, attributeName):
    for attribute in obj.__dict__.items():
        if attribute[0] is attributeName:
            return attribute[1]  # return data for attribute







def rattr(obj, attributeName):
    for attribute in obj.__dict__.items():
        if attribute[0] is attributeName:
            return attribute[1]  # return data for attribute





def mask(eye, img=None, mask=None, dest=None, show=True):
    # print(eye.__dict__.items())
    print("img", img)
    print("mask", mask)
    # m3Show.imshow(img, "mask masked")
    # m3Show.imshow(mask, "to compare")
    # print("eye in mask",eye)
    img = returnAttr(eye, img)
    mask = returnAttr(eye, mask)

    if (img is None or mask is None):
        setattr(eye, dest, np.zeros_like(img))
        m3F.printRed(" IMG OR MASK WAS NONE. NOT MASKING")
        eye.noCircles = True
        return eye
    # print("img.shape", img.shape)
    # print( "mask.shape", mask.shape)
    # destination = returnAttr(eye, dest)
    bah = cv2.bitwise_and(img, mask)
    setattr(eye, dest, bah)
    if show:
        m3Show.imshow(img, "image to be masked")
        m3Show.imshow(mask, "mask")
        m3Show.imshow(bah, "result")

    # print(eye.__dict__.items())

    return eye


# Apply Circ Mask
#  this applies circular mask
def applyCircMask(photo, show=True, useOriginal=True):
    # for photo in photoArray:
    # print(photo)
    for face in photo.faces:
        for eye in face.eyes:
            if (eye.mask is None):
                m3F.printRed("THERE'S NOT MASK")
                break
            # print("eye.image.shape", eye.image.shape)
            # print("eye.mask.shape", eye.mask.shape)
            if useOriginal:
                eye.iris = cv2.bitwise_and(eye.image, eye.mask)
            else:
                eye.iris = cv2.bitwise_and(eye.wip, eye.mask)
            if show:
                m3Show.imshow(eye.iris, "masked")
    return photo

#
#
#
# def placeEyesInFull(photo, fullImgAttr, eyeAttr, dest, show, mask=True):
#     fullImg = None
#     if mask:  # if mask, make black
#         fullImg = np.zeros_like(returnAttr(photo, fullImgAttr))
#     else:
#         fullImg = rattr(photo, fullImgAttr).copy()
#     x, y, channels = fullImg.shape
#
#     for eye in photo.eyes:
#         eyeAttr = rattr(photo, eyeAttr) # get the eyes images to put in full image
#         if eyeAttr is not None:
#             coor = eye.cropRect
#             # fullMask[coor[0]:x, coor[1]:y] = eye.mask
#             # print("coor", coor)
#             fullImg[coor[1]:coor[1]+eyeAttr.shape[0], coor[0]:coor[0]+eyeAttr.shape[1]] = eyeAttr
#     setattr(photo, dest, fullImg)
#     if (show):
#         m3Show.imshow(fullImg, "fullImg")
#
#     return photo

# def cropToEue(photo,):
#     for face in photo.faces:
#         for eye in face:
