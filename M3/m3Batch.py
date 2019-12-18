import numpy as np
import glob
import cv2
import os
from datetime import datetime
from . import m3F
from M3 import m3Class
from M3 import m3Show
from M3 import m3CSV, m3Mask

# **********************************************************************
# exampe of input
'''Batch process a image folder with an array of functions.

Keyword arguments:
inputFolder -- String for inputFolder (relative to where function is called)
functionArray -- A dictionary of function : {params}. see example:

fArray = {function1: {"inputImg": "tobereplacebyactualimageobj", "param1": "blalbal1"},
          function2: {"inputImg": "ignorethis", "param2": "blalbal2"}}

export = boolean on whether to export the final result as an img
'''
# **********************************************************************

#        _           _
#       | |         | |
#  _ __ | |__   ___ | |_ ___
# | '_ \| '_ \ / _ \| __/ _ \
# | |_) | | | | (_) | || (_) |
# | .__/|_| |_|\___/ \__\___/
# | |
# **********************************************************************
# |_|


# takes input of either string for folder w. pics, or path for single.
# takes dictionaries of function: parameters for both
# * pre-processing e.g. find eyes
# * processing of eye images
# * processing of irisArray
# * a post processing of images (e.g. generate comparison)
# debug-variables toggles show parameter for corresponding functions
def photoBatch(ins, functionArray,
               postArray=None, preArray=None, irisArray=None,
               debug=True, debugIris=True):
    photoArray = []
    # **********************************************************************
    #  checking if folder with imgs, or just a single img
    if (type(ins) is type("str")): #is folder, since it's a folder path
        # doing batchprocess for folder of images
        # m3F.printBlue("doing batchprocess for folder of images")
        inputImages = glob.glob(ins + "*.*g")
        inputImages.sort()
        #print("is " + inputImages)


        for imagePath in inputImages:
            # print("?????!!!!!??????")
            inputImage = cv2.imread(imagePath, -1)
            photoArray.append(m3Class.Photo(inputImage, imagePath))
    else:
        # doing batchprocess for single image
        # m3F.printBlue("doing batchprocess for single image")
        photoArray.append(m3Class.Photo(ins, "bla"))

    # **********************************************************************
    # perform PRE functions ( like face detection! )
    print("**********************************************************************")
    print("**********************************************************************")
    m3F.printBlue("DOING PRE FUNCIONS")
    print("**********************************************************************")
    print("**********************************************************************")
    for function in preArray:
        print("function.__name__  in pre: ", function.__name__, function.__dir__)
        if (function.__name__ == "findEyes68"):
            # print("FOUND FINDEYES FUNC")
            for photo in photoArray:
                params = preArray[function]
                photo.faces = function(photo, **params)
        elif (function.__name__ == "fakeEyes" ):
            # print("fakeEyes")
            params = preArray[function]
            photoArray = function(photoArray, **params)
        elif (function.__name__ == "findEyes194"):
            params = preArray[function]
            for photo in photoArray:
                photo = function(photo, **params)
        else:
            # for photo in photoArray:
            params = preArray[function]
            function(photoArray, **params)

    # **********************************************************************
    #  Set debug flag for all functions in functionArray
    # if debug:
    # for function in functionArray:
    #     params = functionArray[function]
    #     params["show"] = debug
    #
    # # if irisDebug:
    # for function in irisArray:
    #     params = irisArray[function]
    #     params["show"] = debugIris

    # **********************************************************************
    #  perform iterations on photos
    print("**********************************************************************")
    print("**********************************************************************")
    m3F.printBlue("DOING FUNCION ARRAY")
    print("**********************************************************************")
    print("**********************************************************************")
    for photo in photoArray:
        # print(photo)
        # for face in photo.faces:
        #     print(face)
        doingFor = None
        # print("functionArray", functionArray)
        for el in functionArray:
            print("el", el, type(el))
            # el = retdict(**el)
            # print("el", el, type(el))
            for function, params in el.items():
                # params = functionArray[function]
                print("function, params", function, params)

                if (function == "doFor"):
                    # print("shitzngiggles!!!", )
                    for face in photo.faces:
                        for eye in face.eyes:
                            doingFor = params["doAs"]
                            setattr(eye, params["doAs"], getattr(eye, params["original"]))
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
                            m3F.printGreen("doingFor")
                            # m3F.printGreen(str(doingFor))
                            if doingFor == "eye":
                                eye = function(eye, **params)
                            else:
                            # print("doingFor",doingFor, "gottenattr", getattr(eye, str(doingFor)))
                                setattr(eye, doingFor, function(getattr(eye, str(doingFor)), **params))
        # photo = iterFunction(photo, functionArray)

    # **********************************************************************
    # perform stuff on irisArray
    print("**********************************************************************")
    print("**********************************************************************")
    m3F.printBlue("DOING IRIS FUNCION ARRAY")
    #          _ . - = - . _
    #        . "  \  \   /  /  " .
    #      ,  \                 /  .
    #    . \   _,.--~=~"~=~--.._   / .
    #   ;  _.-"  / \ !   ! / \  "-._  .
    #  / ,"     / ,` .---. `, \     ". \
    # /.'   `~  |   /:::::\   |  ~`   '.\
    # \`.  `~   |   \:::::/   | ~`  ~ .'/
    #  \ `.  `~ \ `, `~~~' ,` /   ~`.' /
    #   .  "-._  \ / !   ! \ /  _.-"  .
    #    ./    "=~~.._  _..~~=`"    \.
    #      ,/         ""          \,
    #        . _/             \_ .
    #           " - ./. .\. - "
    # print("**********************************************************************")
    # print("**********************************************************************")
    # for function in irisArray:
    #     m3F.printBlue("function name " + function.__name__)
    #     params = irisArray[function]
    #
    #     if ("eye" in params):
    #         # m3F.printBlue(("Doing an eye with" + currentFunctionName))
    #         for photo in photoArray:
    #             for face in photo.faces:
    #                 # if face.eyes is not None:
    #                 for eye in face.eyes:
    #                     params["eye"] = eye
    #                     eye = function(**params)
    #     else:
    #         for photo in photoArray:
    #             for face in photo.faces:
    #                 if face.eyes is not None:
    #                     for eye in face.eyes:
    #                         if eye.iris is not None:
    #                             params["inputImg"] = eye.iris
    #                             eye.iris = function(**params)
    #                             # m3Show.imshow(eye.iris, "eye.iris NIELS TEST")
    # # **********************************************************************
    #  perform POST functions ( such as generate comparison etc.)f
    for function in postArray:
        params = postArray[function]
        if (function.__name__ == "exportToFolder"):
            function(photoArray, ins, **params)
        else:
            photoArray = function(photoArray, **params)
    return photoArray

# **********************************************************************
# **********************************************************************
# **********************************************************************

# This loads our 'truth' masks, since images are properly named, they are assigned using a loop on sorted paths.
# it crops the truth masks for each eye, and their cropping rect.
def loadMasksForComparison(photoArray, maskFolder):
    maskImgs = glob.glob(maskFolder + "*.*g")  # load folder of masked images
    # print("maskImgs", maskImgs)
    maskImgs.sort()  # sort filenames, so 001test comes first, and so on
    # print("maskImgs", maskImgs)
    count = 0
    for photo in photoArray:
        # print("LOADING: ", maskImgs[count], "FOR ", photo.path)

        photo.testMask = cv2.imread(maskImgs[count], -1)
        m3Show.imshow(photo.testMask, "photo.testMask")
        count += 1
        for face in photo.faces:
            for eye in face.eyes:
                eye.testMask = m3F.typeSwap(
                                m3F.typeSwap(photo.testMask)
                                .crop(eye.cropRect))
            # typeswap, as we want
            # to use a pillow func on np-array.
    return photoArray

# **********************************************************************
# **********************************************************************
# **********************************************************************


def iterFunction(photo, functionArray):
    # iter for iterate ,
    # Iterate: make repeated use of a mathematical or computational procedure,
    # applying it each time to the result of the previous application; perform iteration.

    for function in functionArray:
        functionParams = functionArray[function]
        currentFunctionName = function.__name__
        m3F.printBlue("function name " + currentFunctionName)
        # currentFunction["inputImg"] = inputImages
        # **********************************************************************
        if ("inputImg" in functionParams):
            # print("was inputImg")
            # print(photo)
            for face in photo.faces:
                if face.eyes is not None:
                    for eye in face.eyes:
                        m3F.printBlue(("Doing an inputimg as eye.wip with" +    currentFunctionName))
                        functionParams["inputImg"] = eye.wip
                        eye.wip = function(**functionParams)
                        # m3Show.imshow(eye.wip, "eye.wip")
        if ("photo" in functionParams):
            m3F.printBlue("Doing an photo with" + currentFunctionName)
            functionParams["photo"] = photo
            photo = function(**functionParams)
        if ("eye" in functionParams):
            m3F.printBlue(("Doing an eye with" + currentFunctionName))
            for face in photo.faces:
                if face.eyes is not None:
                    for eye in face.eyes:
                        functionParams["eye"] = eye
                        eye = function(**functionParams)
        if ("iris" in functionParams):
            m3F.printBlue(("Doing an eye with" + currentFunctionName))
            for face in photo.faces:
                if face.eyes is not None:
                    for eye in face.eyes:
                        functionParams["eye"] = eye.masked
                        eye.masked = function(**functionParams)
    return photo

# **********************************************************************
# **********************************************************************
# **********************************************************************
# **********************************************************************
# **********************************************************************

#  attrs: a list of attrs of eye. like ["image", "wip", "iris", "mask"]


def export(photoArray,
                       attrs=None,
                       concat=False,
                        folderName=None,
                         exportFullMaskOf=None,
                         CSV=True,
                         settings=None):
    # print("generateComparison
    os.makedirs("EXPORTS/" + folderName + "/", exist_ok=True)
    if exportFullMaskOf:
        photoArray = m3Mask.makeFullMask(photoArray,  exportFullMaskOf)
    for photo in photoArray:
        facesToSave = []
        if (exportFullMaskOf):
            os.makedirs("EXPORTS/" + folderName + "/", exist_ok=True)
            cv2.imwrite("EXPORTS/" + folderName + "/"
                        + os.path.basename(photo.path) + "_FullMask" + ".jpg", photo.fullMask)

        for face in photo.faces:
            if not (type(face.eyes) == type(None)): # TODO: fix this
                for eye in face.eyes:
                    # print(attrs)
                    for attr in eye.__dict__.items():
                        # print("attr", attr)
                        if attr[0] in attrs:
                            # if
                            # print("attr[1].itemsize", attr[1].size)
                            if attr[1].size > 1:
                                # print(attr[0],attr[1], type(attr[1]), attr[1].shape)
                                if concat:
                                    facesToSave.append(attr[1])
                                else:
                                    cv2.imwrite("EXPORTS/" + folderName + "/" + os.path.basename(photo.path) + "_" + attr[0] + ".jpg", attr[1])
                            # else:
                            #     facesToSave.append(eye.image)
            # if (len(eyesToSave) > 1):
                # facesToSave.append(eyesToSave)
            # else:
                # facesToSave.append(eyesToSave[0])
        now = datetime.now()
        now_string = now.strftime("%d-%m-%Y--%H-%M-%S")
        # print(facesToSave[0], type(facesToSave[0]))
        if concat:
            output = concat(facesToSave, direction="v")
            m3Show.imshow(output, "generateComparison output")
            cv2.imwrite("EXPORTS/" + folderName + "/" + os.path.basename(photo.path) + "_" + ".jpg", output)
        # if (folderName is not None):
        #
        # else:

        m3CSV.makeCSV(photoArray, "EXPORTS/" + folderName + "/" + folderName + ".csv")
        file = open("EXPORTS/" + folderName + "/" + "settings.txt","w+")
        count = 1
        # for element in m3F.funcArrToStr():
        #     print("element",element)
        file.write(str(settings))
            # count += 1
        file.close()
    return photoArray
# **********************************************************************
# **********************************************************************

# used to concatenate pictures (used in generateComparison).
# i.e. putting pictures besides eachother, or below
# first dimensions are sorted, as the dim must match for the side they're
# concatenated.

def concat(images, direction="h"):
    # print("images", images)
    hs, ws = [], []  # used for storing widths and heights,
                     # as images should be of the same size in one of the directons for np.concatenate to work.
                     # used for sorting, and picking highest value
    for img in images:
        # print("img!!!!", img)
        if (len(img.shape) == 3):
            h, w, c = img.shape
            hs.append(h)
            ws.append(w)
        elif(len(img.shape) == 2):
            h, w = img.shape
            hs.append(h)
            ws.append(w)
        # else:
            # images.remove(img)
    hs.sort(reverse=True)
    ws.sort(reverse=True)
    outImgs = []
    for img in images:
        if img is not None:
            # print("img", img)
            newSize = np.zeros_like(img)
            if (len(img.shape) == 3):
                newSize = np.resize(newSize, (hs[0], ws[0], 3))
            elif(len(img.shape) == 2):
                newSize = np.resize(newSize, (hs[0], ws[0]))
            newSize[0:img.shape[0], 0:img.shape[1]] = img
            outImgs.append(newSize)
    # print("ooutImgs", outImgs)
    # for img in outImgs:
        # m3Show.imshow(img, "outimg in outimages (concat)")
    if (direction == "h"):
        result = np.concatenate(outImgs, axis=1)
    else:
        result = np.concatenate(outImgs, axis=0)
    return result


# **********************************************************************
# **********************************************************************
# **********************************************************************
# **********************************************************************

# makes face objs with a single eye obj ( when running processing on exported eyes)
def fakeEyes(photoArray):
    print("fakeEyes ran")
    for photo in photoArray:
        # m3Show.imshow(photo.originalImage,"asdfasdf")
        temp = []
        temp.append(m3Class.Face([m3Class.Eye(photo.originalImage)]))
        photo.faces = temp
    return photoArray
