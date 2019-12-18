import cv2
import numpy as np
from matplotlib import pyplot as plt
import PIL
import cmath
from PIL import ImageFilter, ImageEnhance
from M3 import m3Show
from M3 import m3Class
from M3 import m3F as m3F
import sys
sys.path.append("/M3")


def semicircle(eye, show):
    #path = "output_eyes/photo19_Face_1_Left.jpeg"
    #iris = "iris_test2.jpeg"
    #img = cv2.imread(path)
    
    #img = eye.wip
    img = eye.iris
    cimg = img
    m3F.imshow(cimg, "cimg")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(img,100,img.shape[0])
    
    m3F.imshow(canny, "semi circle canny")
    
    
    for i in eye.circle[0,:]:
        center = (i[0],i[1])
        radius = i[2]
        #cv2.circle(cimg, center, 3, (0,255,255), 0)
        #cv2.circle(cimg, center, radius, (0,0,255), 0)


    dt = cv2.distanceTransform(255-canny, cv2.DIST_L2, 5)

    minInlierDist = 2.0
    
    for i in eye.circle[0,:]:
        counter = 0
        inlier = 0
        center = (i[0],i[1])
        radius = i[2]

        maxInlierDist = radius/200.0
        if (maxInlierDist < minInlierDist):
            maxInlierDist = minInlierDist

        t = 0
        while t < 2*cmath.pi:
            counter += 1
            cX = int(radius*cmath.cos(t) + i[0])
            cY = int(radius*cmath.sin(t) + i[1])
            try:
                if (dt[cY,cX] < maxInlierDist and gray[cY,cX] > 100):
                    inlier += 1
                    cv2.circle(cimg, (cX,cY), 3, (255,0,0))
            except IndexError:
                print("skipped")



            t += 0.1



    finalImg = cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB)
    if (show):
        m3F.imshow(finalImg, "semicircles")
    #return finalImg
    return eye
    
    
    #plt.imshow((img * 255).astype(np.uint8))
    #plt.show

