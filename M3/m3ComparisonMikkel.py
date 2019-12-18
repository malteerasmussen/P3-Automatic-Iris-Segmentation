import cv2
import PIL
import numpy as np
from M3 import m3Show
from M3 import m3CSV
from M3 import m3F
#INPUTIMAGE 2 IS THE BASECASE - The handmask

# inputImg1 = cv2.imread("15.jpg_0hardedge.jpg",cv2.IMREAD_GRAYSCALE)
# inputImg2 = cv2.imread("15.jpg_0softedge.jpg",cv2.IMREAD_GRAYSCALE)
#
def batchcc(photoArray):
    for photo in photoArray:
        for face in photo.faces:
            for eye in face.eyes:
                ret, thresh1 = cv2.threshold(eye.iris,1,255,cv2.THRESH_BINARY)
                m3Show.imshow( eye.testMask, "MANMADE:")
                m3Show.imshow(thresh1, "thresh1:")
                compare(eye.testMask,thresh1)
    return photoArray

def compare(machineMask,manMask):

    print (machineMask)
    print (manMask)

    inputImg1=cv2.cvtColor(machineMask, cv2.COLOR_BGR2GRAY)
    inputImg2=cv2.cvtColor(manMask, cv2.COLOR_BGR2GRAY)
    print ("machineMask",machineMask)
    print ("manMask",manMask)
    print( inputImg1.shape, "inputImg1", inputImg2.shape, "inputImg2")
    print (machineMask.shape)
    print (manMask.shape)



    rows1,cols1= inputImg1.shape
    count1 = 0
    for i in range(rows1):
        for j in range(cols1):
            count1 += inputImg1[i,j]

    count1 = count1/255
    print(count1)

    rows2,cols2 = inputImg2.shape
    #Pixel-count
    cntPixels2 = rows2*cols2

    count2 = 0
    for i in range(rows2):
        for j in range(cols2):
            count2 += inputImg2[i,j]
    count2 = count2/255
    print(count2)

    deviatingValuedPixelPer = (abs(count1-count2)/count2)*100
    overlappingValuedPixelPer = 100-deviatingValuedPixelPer



    print ("Percentage overlapping of valued pixels from BASECASE",overlappingValuedPixelPer)


import cv2
from matplotlib import pyplot as plt
import numpy as np
from M3 import m3Show


def pixelcomparison(photoArray, eyeAttr="", show=True):
    for photo in photoArray:
        for face in photo.faces:
            for eye in face.eyes:
                if eye.noCircles is True:
                    eye.TP = 0
                    eye.FN = 0
                    eye.FP = 0
                    eye.TPFN = 0
                else:

                    # ret, autoThresh = cv2.threshold(eye.iris,1,255,cv2.THRESH_BINARY)
                    orgAutoMask = getattr(eye, eyeAttr)
                    # ret, orgHandMask = cv2.threshold(eye.testMask,1,255,cv2.THRESH_BINARY)
                    orgHandMask = eye.testMask
                    # ret, orgHandMask = cv2.threshold(orgHandMask,1,255,cv2.THRESH_BINARY)
                    # print("orgHandMask", eye.testMask)
                    # print( orgAutoMask.shape, "orgAutoMask", orgHandMask.shape, "orgHandMask")
                    # print("orgHandMask", eye.testMask)
                    # orgAutoMask=cv2.cvtColor(orgAutoMask, cv2.COLOR_BGR2GRAY)
                    # orgHandMask=cv2.cvtColor(orgHandMask, cv2.COLOR_BGR2GRAY)
                    # orgHandMask = m3F.getRed(orgHandMask, False)
                    # orgAutoMask = m3F.getRed(orgAutoMask, False)
                    # m3Show.imshow(orgAutoMask, "orgAutoMask")
                    # m3Show.imshow(orgHandMask, "orgHandMask")
                    # m3Show.imshow("orgAutoMask", orgAutoMask, gray=True)
                    # m3Show.imshow("orgHandMask",orgHandMask, gray=True)

                    # m3Show.imshow(orgHandMask,"handMask")

                    height,width,c  = orgHandMask.shape
                    imgSize=height*width
                    TruePositive = 0.0
                    FalsePositive = 0.0
                    FalseNegative = 0.0
                    handMaskAccumLum = 0.0
                    autoMaskAccumLum = 0.0

                    autoMask = orgAutoMask.astype("float64")
                    handMask = orgHandMask.astype("float64")
                    print( autoMask.shape, "autoMask", handMask.shape, "handMask")
                    for y in range(height):
                        for x in range(width):

                            autoMask[y,x,0]= (orgAutoMask[y,x,0]/255)
                            handMask[y,x,0]= (orgHandMask[y,x,0]/255)
                            # print("autoMask[y,x,0]", autoMask[y,x,0],"handMask[y,x,0]",handMask[y,x,0])
                            if (autoMask[y,x,0] == handMask[y,x,0]):
                                TruePositive += handMask[y,x,0]

                            if (autoMask[y,x,0] < handMask[y,x,0]):
                                FalseNegative += (handMask[y,x,0]-autoMask[y,x,0])
                                TruePositive += autoMask[y,x,0]

                            if (autoMask[y,x,0] > handMask[y,x,0]):
                                FalsePositive += autoMask[y,x,0] - handMask[y,x,0]
                                TruePositive += handMask[y,x,0]

                            handMaskAccumLum += handMask[y,x,0]
                            autoMaskAccumLum += autoMask[y,x,0]

                    # print(autoMask)
                    #
                    print ("handMaskAccumLum",handMaskAccumLum)

                    print("TruePositive",TruePositive)
                    print("FalseNegative",FalseNegative)
                    print("FalsePositive",FalsePositive)

                    TruePositive= (TruePositive/handMaskAccumLum)*100
                    FalseNegative=(FalseNegative/handMaskAccumLum) *100
                    FalsePositive1=(FalsePositive/autoMaskAccumLum)*100
                    FalsePositive=(FalsePositive/handMaskAccumLum)*100
                    eye.TP = TruePositive
                    eye.FN = FalseNegative
                    eye.FP = FalsePositive
                    eye.TPFN = FalseNegative + TruePositive
                    # eye.TP = TruePositive
                    print("TruePositive%",TruePositive,"% of accumulated pixel values of handmask")
                    print("FalseNegative%",FalseNegative,"% of accumulated pixel values of handmask")
                    print("FalsePositive%",FalsePositive,"% of accumulated pixel values of handmask")
                    print("FalsePositive%",FalsePositive1,"% of accumulated pixel values of automask")


                    autoMask = autoMask*255
                    handMask = handMask*255

                    autoMask = orgAutoMask.astype("uint8")
                    handMask = orgHandMask.astype("uint8")

                # M3Show.imshow(orgHandMask,"handMask")
                # M3Show.imshow(orgAutoMask,"autoMask")
    return photoArray
