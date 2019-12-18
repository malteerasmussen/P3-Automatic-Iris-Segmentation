import numpy as np
import glob
import cv2
import os
from datetime import datetime
from . import m3F
from M3 import m3Class
from M3 import m3Show
from M3 import m3CSV
from copy import deepcopy

class photoBatcher:
    photoArray = None

    def __init__(self,inputFolder, preArray):
        print("init_ran", inputFolder, preArray)
        # **********************************************************************
        # import folder of images, append as photo obj to photoArray
        # ______________________________________________________________________
        inputImages = glob.glob(inputFolder + "*.*g")
        inputImages.sort()
        #print("is " + inputImages)
        self.photoArray = []
        for imagePath in inputImages:
            # print("?????!!!!!??????")
            print("imagePath", imagePath)
            inputImage = cv2.imread(imagePath, -1)
            self.photoArray.append(m3Class.Photo(inputImage, imagePath))
        print("photoArray", self.photoArray)
        # **********************************************************************
        # **********************************************************************
        for function in preArray:
            # print("function.__name__  in pre: ", function.__name__, function.__dir__)
            if (function.__name__ == "findEyes68"):
                # print("FOUND FINDEYES FUNC")
                for photo in self.photoArray:
                    params = preArray[function]
                    photo.faces = function(photo, **params)
            elif (function.__name__ == "findEyes194"):
                params = preArray[function]
                for photo in self.photoArray:
                    photo = function(photo, **params)
            else:
                # for photo in photoArray:
                params = preArray[function]
                self.photoArray = function(self.photoArray, **params)

    # **********************************************************************
    paCopy = None
    def iterate(self, functionArray):
        print("iterate() photoArray", self.photoArray)
        self.paCopy = deepcopy(self.photoArray)#[:]
        for photo in self.paCopy:
            doingFor = None
            # print("functionArray", functionArray)
            for el in functionArray:
                # print("el", el, type(el))
                # el = retdict(**el)
                # print("el", el, type(el))
                for function, params in el.items():
                    # params = functionArray[function]
                    # print("function, params", function, params)

                    if (function == "doFor"):
                        # print("shitzngiggles!!!", )
                        doingFor = params["doAs"]
                        for face in photo.faces:
                            for eye in face.eyes:
                                setattr(eye, params["doAs"], getattr(eye, params["original"]))
                                # cv2.imwrite("EXPORTS/COMPARISONS/" + now_string + ".jpg", output)
                                saveStr = "EXPORTS/COMPARISONS/" + "didAs" + params["doAs"] + ".jpg"
                                print("saveStr", saveStr )
                                cv2.imwrite(saveStr, getattr(eye, params["original"]))
                            # m3Show.imshow(eye.image, "orignal")
                            # m3Show.imshow(getattr(eye, params["doAs"]), params["doAs"])
                            # print(eye.__dict__.items())
                    elif (function == "doForEye"):
                        print("doingFor = eye")
                        doingFor = "eye"
                    else:

                        print("function.__name__  in pre: ", function.__name__)
                        for face in photo.faces:
                            for eye in face.eyes:
                                print("!!!!!!!!doingFor", doingFor)
                                # m3F.printGreen(str(doingFor))
                                if doingFor == "eye":
                                    eye = function(eye, **params)
                                else:
                                    if not eye.noCircles:
                                # print("doingFor",doingFor, "gottenattr", getattr(eye, str(doingFor)))
                                        result =  function(getattr(eye, str(doingFor)), **params)
                                        setattr(eye, doingFor, result)
                                        saveStr = "EXPORTS/COMPARISONS/" + "didAs" + doingFor + function.__name__+ ".jpg"
                                        print("saveStr", saveStr )
                                        cv2.imwrite(saveStr, result)

                                        # cv2.imwrite("EXPORTS/COMPARISONS/mask194", eye.outline)
    def post(self, postArray):
        for function in postArray:
            params = postArray[function]
            self.paCopy = function(self.paCopy, **params)
