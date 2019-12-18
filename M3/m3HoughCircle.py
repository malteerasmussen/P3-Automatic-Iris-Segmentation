import cv2, sys
import numpy as np
from matplotlib import pyplot as plt
sys.path.append("/M3")
from M3 import m3F as m3F
from M3 import m3Class, m3Show, m3Mask
import os
from PIL import ImageFilter, ImageEnhance
import PIL

def findCircleAndMakeMask(eye, eyeAttr="", houghParams="", maskParams="", show=True):
    image = getattr(eye, eyeAttr)
    eye.noCircles = False
    eye.circles = False
    eye.houghMask = None
    circles = findCircle(image, **houghParams)
    eye.circles = circles
    print("circles", circles)
    eye.outline = m3Mask.makeCircularOutline(eye, circles)
    if circles is not None:
        eye.houghMask = makeCircularMask(eye.image, circles, show=True, onlyOne=True)
    else:
        eye.noCircles = True
        eye.circles = None
    return eye


# TODO - RENAME THEM FROM PARAMX TO SOOMETHINNG MEANINGFUL
def findCircle(img, resolution, min_dist, param_1, param_2, min_radius_width_divider, max_radius_width_divider, show):
    # old params for HoughCircle: img, cv2.HOUGH_GRADIENT, 1.5, 120, param1=60, param2=15, minRadius=0, maxRadius=int(m3F.typeSwap(img).height / 2))
    min_radius = int(m3F.typeSwap(img).width/min_radius_width_divider)
    max_radius = int(m3F.typeSwap(img).width/max_radius_width_divider)
    return run(img, resolution, min_dist, param_1, param_2, min_radius, max_radius, show)


def findCircleSimple(eye, show):
    #run(eye, 1, 120, 60, 15, 10, 100, show)
    # print("eye type", type(eye))
    if isinstance(eye, type(m3Class.Eye())):
        run(eye, 1, 120, 200, 10, int(m3F.typeSwap(eye.wip).width/4), int(m3F.typeSwap(eye.wip).width/3), show)
    else:
        run(eye, 1, 120, 200, 10, int(m3F.typeSwap(eye).width/4), int(m3F.typeSwap(eye).width/3), show)

def findCircleSimpleEdge(eye, show):
    #run(eye, 1, 120, 60, 15, 10, 100, show)
    # print("eye type", type(eye))
    if isinstance(eye, type(m3Class.Eye())):
        run(eye, 1, 120, 200, 10, int(m3F.typeSwap(eye.wip).width), 0, show)
    else:
        run(eye, 1, 120, 200, 10, int(m3F.typeSwap(eye).width), 0, show)


def findCircleDouble(eye, resolution, min_dist, param_1, param_2, min_radius, max_radius, show):
    # old params for HoughCircle: img, cv2.HOUGH_GRADIENT, 1.5, 120, param1=60, param2=15, minRadius=0, maxRadius=int(m3F.typeSwap(img).height / 2))
    runDouble(eye, resolution, min_dist, param_1, param_2, min_radius, max_radius, show)

def findCircleSimpleDouble(eye, show):
    #run(eye, 1, 120, 60, 15, 10, 100, show)
    runDouble(eye,1,120,200,10,int(m3F.typeSwap(eye).height/6),int(m3F.typeSwap(eye).height/2.5),show)


# TODO: MAKE NOTES FOR THIS
def run(tempeye, tempResolution, tempMin_dist, tempParam_1, tempParam_2, tempMinRadius, tempMaxRadius, tempShow):
    print("***************************************************************************************")

    img = tempeye.copy()
    print(tempMinRadius)
    # isEyeClass = False
    # eye = None
    # if isinstance(img, type(m3Class.Eye())):
    #     isEyeClass = True
    #     eye = tempeye
    #     img = eye.wip.copy()
        # print("DID EYE")
    if not isinstance(img, type(None)):
        cimg = img.copy()

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # print("Hough run : min:", tempMinRadius, " max: ", tempMaxRadius)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, tempResolution, tempMin_dist,
                                   param1=tempParam_1, param2=tempParam_2, minRadius=tempMinRadius, maxRadius=tempMaxRadius)

#        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,,120,param1=200,param2=10,
#                                   minRadius=int(m3F.typeSwap(img).height/6),maxRadius=int(m3F.typeSwap(img).height/2.5))
        # print("circles!!!!", circles)
        # return circles
        if not isinstance(circles, type(None)):

            if (circles.size != 0):
                circles = np.uint16(np.around(circles))
                # print(circles)
                index = 0
                if (tempShow):
                    for i in circles[0, :]:
                    # if img[i[1],i[0]] < 15:
                        # draw the outer circle
                            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
                        # draw the center of the circle
                            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
                # eye.houghOutline = cimg

                    m3F.imshow(cimg, "Circle")
                    m3F.printGreen("CIRCLES FOUND^^^")
                    print("img out", img.shape)
            return circles
        else:
            # eye.noCircles = True
            if (tempShow):
                m3F.imshow(cimg, "no circles found")
                m3F.printRed("NO CIRCLES FOUND^^^")
            return None
    else:
        m3F.printRed("Image is NONE")


def runDouble(tempeye, tempResolution, tempMin_dist, tempParam_1, tempParam_2, tempMinRadius, tempMaxRadius, tempShow):
    # print("***************************************************************************************")

    img = tempeye
    # print(tempMinRadius)

    if not isinstance(img, type(None)):
        cimg = img
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        bwimg = img.copy()

        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, tempResolution, tempMin_dist,
                                   param1=tempParam_1, param2=tempParam_2, minRadius=tempMinRadius, maxRadius=tempMaxRadius)

        if not isinstance(circles, type(None)):

            if (circles.size != 0):
                circles = np.uint16(np.around(circles))
                # print(circles)

                for i in circles[0, :]:
                    bwimg.fill(0)
                    # draw the outer circle
                    cv2.circle(bwimg, (i[0], i[1]), i[2], 255, -1)
                    # draw the center of the circle
                    # cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
                    # m3F.imshow(bwimg,"circle")

                    mask = cv2.bitwise_and(img, bwimg)
                    # finalImg = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

                    #m3F.imshow(mask, "final img")

                    pupilCircle = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 50,
                                                   param1=200, param2=10, minRadius=int(i[2]/6),
                                                   maxRadius=int(i[2]/4*3))
                    if not isinstance(pupilCircle, type(None)):

                        if (pupilCircle.size != 0):
                            pupilCircle = np.uint16(np.around(pupilCircle))
                            # print(circles)
                            i = 1
                            for j in pupilCircle[0, :]:
                                wiggle = 5
                                if i[0] + wiggle > j[0] > i[0] - wiggle and i[1] + wiggle > j[1] > i[1] - wiggle:
                                    # draw the outer circle
                                    count = 255/i
                                    i += 1
                                    cv2.circle(mask, (j[0], j[1]), j[2], (count, count, 0), 1)
                                    # draw the center of the circle

                                    cv2.circle(mask, (j[0], j[1]), 2, (0, 0, 255), 3)
                            m3F.imshow(mask, "final img")

            if (tempShow):
                m3F.imshow(cimg, "Circle")
                m3F.printGreen("CIRCLES FOUND^^^")
                print("img out", img.shape)
            return img
        else:

            if (tempShow):
                m3F.imshow(cimg, "no circles found")
                m3F.printRed("NO CIRCLES FOUND^^^")
    else:
        m3F.printRed("Image is NONE")



def makeCircularMask(photo, show=True, onlyOne=True):
    # for photo in photoArray:
        # print("photo")
    # photo = inputImg
    for face in photo.faces:
        # print("facee")
        for eye in face.eyes:
            # print("eye")
            maskImg = np.zeros_like(eye.wip)
            if not isinstance(eye.circle, type(None)):
                # firstCircle = eye.circle[0]
                if onlyOne:
                    i = eye.circle[0][0]
                    # print("i", i)
                    # print("onlyone", i[0], i[1], i[2])
                    cv2.circle(maskImg, (i[0], i[1]), i[2], (255,  255, 255), -1)
                else:
                    for i in eye.circle[0, :]:
                        # print("i", i)
                        cv2.circle(maskImg, (i[0], i[1]), i[2], (255,  255, 255), -1)
                if (show):
                    m3Show.imshow(maskImg, "mask")
                eye.mask = maskImg
    return photo

def makeCircularMask(img, circles, show=True, onlyOne=True):
    # for photo in photoArray:
        # print("photo")
    # photo = inputImg
            # print("eye")
    maskImg = np.zeros_like(img)
    # if not isinstance(eye.circle, type(None)):
        # firstCircle = eye.circle[0]
    if circles is not None:
        if onlyOne:
            i = circles[0][0]
            # print("i", i)
            # print("onlyone", i[0], i[1], i[2])
            cv2.circle(maskImg, (i[0], i[1]), i[2], (255,  255, 255), -1)
        else:
            for i in circles[0, :]:
                # print("i", i)
                cv2.circle(maskImg, (i[0], i[1]), i[2], (255,  255, 255), -1)
    else:
        return maskImg
    if (show):
        m3Show.imshow(maskImg, "mask")
    return maskImg
