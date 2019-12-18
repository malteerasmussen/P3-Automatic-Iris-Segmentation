import json

import cv2
from matplotlib import pyplot as plt
from M3 import m3F as m3F
import numpy as np
import Retinex
from Retinex import retinex




def runMSRCP(inputImg, show):
    with open('Retinex/config.json', 'r') as f:
      config = json.load(f)

    #img = cv2.imread(imgPath,1)


    img_msrcp = retinex.MSRCP(
        inputImg,
        config['sigma_list'],
        config['low_clip'],
        config['high_clip']
    )

    if (show):
        m3F.imshow(img_msrcp, "MSRCP")

    return img_msrcp

def runSSR(inputImg, sigma, show):

    img_ssr = retinex.singleScaleRetinex(inputImg, sigma)

    plt.imshow(img_ssr)
    plt.show

    if (show):
        m3F.imshow(img_ssr, "SSR")

    return img_ssr
