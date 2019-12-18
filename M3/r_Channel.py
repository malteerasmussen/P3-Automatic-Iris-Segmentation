import cv2

def r_Channel (inputImg):
   

    b = inputImg.copy()
    # set green and red channels to 0
    b[:, :, 1] = 0
    b[:, :, 2] = 0


    g = inputImg.copy()
    # set blue and red channels to 0
    g[:, :, 0] = 0
    g[:, :, 2] = 0

    r = inputImg.copy()
    # set blue and green channels to 0
    r[:, :, 0] = 0
    r[:, :, 1] = 0


    # RGB - Blue
    cv2.imshow('B-RGB', b)

    # RGB - Green
    cv2.imshow('G-RGB', g)

    # RGB - Red
    redImg= cv2.imshow('R-RGB', r)
    
    return (redImg)