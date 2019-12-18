from M3 import m3Class
from M3 import m3F as m3F
import cv2
import sys
from PIL import Image, ImageDraw
import face_recognition
import numpy as np
from matplotlib import pyplot as plt
sys.path.append("/M3")
from M3 import m3F as m3F
from M3 import m3Show
from datetime import datetime
import dlib

# this uses the library face_recognition to find 68 points.
# first the image is downscaled, to allow faster processing
# the landmarks of each are are sorted, as to get the boundaries of the eye (ex. min x equals left edge)
# A margin of 0.2 is added
# face obj with eyes, cropping rectangle, and the landmarks is returned

def findEyes68(photo, division, show=True, debug=False):
    h, w, c = photo.originalImage.shape
    downScaledDim = ((round(w/division)), round(h/division))
    copy = photo.originalImage.copy()
    pil_image = Image.fromarray(copy)
    inputImg = copy
    inputImg = cv2.resize(inputImg, downScaledDim)
    photo.loResImage = inputImg

    if debug:
        print("originalShape", h, w, c)
        print("downScaledDim", downScaledDim)
        m3Show.imshow(m3F.typeSwap(pil_image), "image to find eyes on")

    face_landmarks_list = face_recognition.face_landmarks(inputImg)

    if debug:
        print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))
        print("face_landmarks_list type was", type(face_landmarks_list))

    if (len(face_landmarks_list) == 0):

        m3F.printRed(" found no faces in this picture")
        plt.imshow(cv2.cvtColor(inputImg, cv2.COLOR_RGB2BGR))
        plt.show()
        return [m3Class.Face(noFaceImg=inputImg)]

    faces = []
    for face_landmarks in face_landmarks_list:

        LeftEyeLandmarks = face_landmarks['left_eye']
        rightEyeLandmarks = face_landmarks['right_eye']

        # imgwithPoints = inputImg.copy()
        # for eye in (LeftEyeLandmarks, rightEyeLandmarks):
        #     for x, y in eye:
        #         imgwithPoints = cv2.rectangle(imgwithPoints, (x - 20, y - 20), (x + 20, y + 20), (0, 0, 255), -1)
        # **********************************************************************
        # if debug:
        #     m3Show.imshow(imgwithPoints, "with points")
        eyes = []
        for eyeLandmarkPoints in LeftEyeLandmarks, rightEyeLandmarks:
            xs, ys = [], []
            for x, y in eyeLandmarkPoints:
                xs.append(x)
                ys.append(y)
                # print("x y ", x, y)
            xs.sort(reverse=True)
            ys.sort(reverse=True)
            maxX = xs[0]
            maxY = ys[0]
            xs.sort(reverse=False)
            ys.sort(reverse=False)
            minX = xs[0]
            minY = ys[0]
            # print("maxX,maxY,minX,minY", maxX, maxY, minX, minY)
            # **********************************************************************
            margin = round((maxX - minX) * 0.2   )
            cropRect = (minX - margin, minY - margin,
                        maxX + margin, maxY + margin)
            cropRectNoMargin = (minX , minY ,
                        maxX, maxY )
            # **********************************************************************
            # margin = round((maxX - minX) * 0.2)
            # cropRect = (minX, minY,
            #             maxX, maxY)
            # **********************************************************************
            cropRect = [round(division*num) for num in cropRect]
            # for num in margin: num * division
            if show:
                m3Show.imshow(np.asarray(pil_image.crop(cropRect)), "cropRect")
            upscaledLandmarks = []
            # croprect: "The box is a 4-tuple defining the left, upper, right, and lower pixel coordinate"
            croppedEye = np.asarray(pil_image.crop(cropRect))
            # print("croppedEye.shape", croppedEye.shape)
            # print("cropRect", cropRect, [round(num/division) for num in cropRect])

            for x, y in eyeLandmarkPoints:
                upscaledLandmarks.append([((x - (minX-margin)) * division), ((y-(minY-margin)) * division)])


                # upscaledLandmarks.append([x * division, y *division])
                # print("minX, x, minY, y, division", (minX, x, minY, y, division, margin))
                #
                # x = x * division
                # y = y * division
                # left = cropRect[0] + margin
                # top = cropRect[1] + margin
                # upscaledLandmarks.append((x-left)-(y-top))
                # y = y * division
            # print("upscaledLandmarks", upscaledLandmarks)
            e = m3Class.Eye(croppedEye, cropRect, upscaledLandmarks)
            print(upscaledLandmarks)
            e.mask68 = makePolyMask(croppedEye, upscaledLandmarks)
            # cv2.imwrite("EXPORTS/COMPARISONS/mask68.jpg", e.mask68)

            e.margin = margin
            e.CRnoMargin = cropRectNoMargin
            e.minX = minX
            e.minY = minY
            eyes.append(e)
        faces.append(m3Class.Face(eyes))
    return faces



# predictor = dlib.shape_predictor(("MODELS/2.dat"))

# def find194(img):
predictor = dlib.shape_predictor("MODELS/shape_predictor_194_face_landmarks.dat")

def findEyes194(photo, division, show=True, ):
    img = photo.loResImage
    detector = dlib.get_frontal_face_detector()
    foundFaces = detector(img, 1)
    # print("foundFaces", foundFaces)
    # print("Number of faces detected: {}".format(len(foundFaces)))
    for index, data in enumerate(foundFaces):
        if index == 0:
            shape = predictor(img, data)
            e0points = []
            e1points = []
            for i in range(shape.num_parts):
                point = shape.part(i)

                # if (i >= 48 and i <= 69):  #
                if (i >= 40 and i <= 75):  #
                    e1points.append([round(point.x ), round(point.y )])
                if (i >= 20 and i <= 50):  #
                    e0points.append([round(point.x ), round(point.y )])
            # print("points", points)
            eyePoints = [e1points,e0points]
            eyeCount = 0
            for eye in photo.faces[0].eyes:
                # print("eyeCount",eyeCount)
                cropRect = eye.cropRect
                # print("cropRect", cropRect)
                # eye.manyLandmarkPoints = []
                points = eyePoints[eyeCount]
                # ys = []
                # print("points", points)
                margin = eye.margin
                upscaledLandmarks = []
                m3Show.imshow(eye.image,"eye.image")
                for x, y in points:
                    # ys.append(y)
                    # ensure that the points are within the croprect from 68 landmarks
                    # croprect: "The box is a 4-tuple defining the left, upper, right, and lower pixel coordinate"

                    if (x * division > cropRect[0] and x * division < cropRect[2]):
                        if (y * division > cropRect[1] and y * division< cropRect[3]):

                    # upscaledLandmarks.append([((x - (cropRect[0]-(margin*division))) ), ((y-(cropRect[1]-(margin*division))) )])
                            # print("cropRect", eye.cropRect)
                    # upscaledLandmarks.append([(((x * division)- ((cropRect[0])  - 0))) , ((((y*division) -((cropRect[1]) -  0)))) ])
                            # eye.manyLandmarkPoints.append([x, y])
                    # upscaledLandmarks.append([((x - (cropRect[0]-margin)) * division), ((y-(cropRect[1]-margin)) * division)])
                            upscaledLandmarks.append([((x - (eye.minX-eye.margin)) * division), ((y-(eye.minY-eye.margin)) * division)])
                # print("194upscaledLandmarks",upscaledLandmarks)
                # eye.mask194 = makePolyMask(photo.originalImage, upscaledLandmarks)
                eye.mask194 = makePolyMask(eye.image, upscaledLandmarks)

                # cv2.imwrite("EXPORTS/COMPARISONS/mask194.jpg", eye.mask194)
                eyeCount += 1
    return photo


# **********************************************************************

def makePolyMask(photoForDims, polys, show=True):
    # polyMask =  np.zeros_like(photo.originalImage)
    polyMask =  np.zeros_like(photoForDims)
    # print("polyMask.shape",polyMask.shape)
    # print("EYE COOR", eye.landmarkPoints)
    # print("np.int_(polys)", np.int_([polys]))
    polyMask = cv2.fillPoly(polyMask, np.int_([polys]), (255, 255, 255))
    # polymask = cv2.fillConvexPoly(polyMask, np.int_([eye.landmarkPoints]), (255, 255, 255),1000)
    # polymask = cv2.drawContours(polyMask, np.int_([eye.landmarkPoints]),-1, (255, 255, 255), 2)
    # polymask = cv2.approxPolyDP(np.int_([eye.landmarkPoints]),4, True)
    # xs, ys = [], []
    # for x, y in eye.eye.landmarkPoints:
    #     xs.append(x)
    #     ys.append(y)
    # polymask = cv2.poly
    # print(polymask)
    if show:
        m3Show.imshow(polyMask, "POLYMASK")
    # epm = m3Class.Eye(np.asarray(pil_image.crop(left)), left, lEyeCoor)
    # epm = polyMask.crop(eye.cropRect)
    # eye.polyMask = m3F.typeSwap(m3F.typeSwap(polyMask).crop(eye.cropRect))

    return polyMask
    # eye.polyMask = polyMask
