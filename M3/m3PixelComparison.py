import cv2
from matplotlib import pyplot as plt
import numpy as np
from M3 import m3Show


def pixelcomparison(inputImg, inputImg2, show):

    orgHandMask = inputImg
    orgAutoMask = inputImg2

    m3Show.imshow(orgHandMask,"handMask")

    height,width= orgHandMask.shape
    imgSize=height*width 
    TruePositive = 0.0 
    FalsePositive = 0.0
    FalseNegative = 0.0
    handMaskAccumLum = 0.0
    autoMaskAccumLum = 0.0

    autoMask = orgAutoMask.astype("float64")
    handMask = orgHandMask.astype("float64")

    for y in range(height):
        for x in range(width):

            autoMask[y,x]= (orgAutoMask[y,x]/255)
            handMask[y,x]= (orgHandMask[y,x]/255)

            if (autoMask[y,x] == handMask[y,x]):
                TruePositive += handMask[y,x]

            if (autoMask[y,x] < handMask[y,x]):
                FalseNegative += (handMask[y,x]-autoMask[y,x])
                TruePositive += autoMask[y,x]/handMask[y,x]

            if (autoMask[y,x] > handMask[y,x]):
                FalsePositive += autoMask[y,x] - handMask[y,x]
                TruePositive += handMask[y,x]/autoMask[y,x]

            handMaskAccumLum += handMask[y,x]
            autoMaskAccumLum += autoMask[y,x]        

    print(autoMask)          

    print ("handMaskAccumLum",handMaskAccumLum)

    print("TruePositive",TruePositive)
    print("FalseNegative",FalseNegative)
    print("FalsePositive",FalsePositive)

    TruePositive= (TruePositive/handMaskAccumLum)*100
    FalseNegative=(FalseNegative/handMaskAccumLum) *100
    FalsePositive1=(FalsePositive/autoMaskAccumLum)*100
    FalsePositive=(FalsePositive/handMaskAccumLum)*100

    print("TruePositive%",TruePositive,"% of accumulated pixel values of handmask")
    print("FalseNegative%",FalseNegative,"% of accumulated pixel values of handmask")
    print("FalsePositive%",FalsePositive,"% of accumulated pixel values of handmask")
    print("FalsePositive%",FalsePositive1,"% of accumulated pixel values of automask")


    autoMask = autoMask*255
    handMask = handMask*255

    autoMask = orgAutoMask.astype("uint8")
    handMask = orgHandMask.astype("uint8")

    M3Show.imshow(orgHandMask,"handMask")
    M3Show.imshow(orgAutoMask,"autoMask")
    
