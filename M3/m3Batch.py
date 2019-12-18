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
