import numpy as np
import glob
import cv2
import os
from datetime import datetime
from . import m3F
from M3 import m3Class
from M3 import m3Show



#


# def exportToFolder(photoArray, inputFolder, parent=None, fileName=None):
#     # outputFolder = os.getcwd() + "/EXPORTS/" + folderName + "_" +  now_string
#     now_string = datetime.now().strftime("-%d-%m-%Y---%H-%M-%S")
#     outputFolder = os.getcwd() + "/EXPORTS/" + inputFolder.replace("PICTURES/InputPictures", "").replace("/", "") + now_string + "/"
#     print("outputFolder", outputFolder)
#     # print("path", outputFolder)
#     if parent is "Photo":
#         for photo in photoArray:
#             pass
#             # rr = photo.__dict__.items()
#     if parent is "Eye":
#         for photo in photoArray:
#             for face in photo.faces:
#                 eyeCount = 0
#                 for eye in face.eyes:
#                     for attr in eye.__dict__.items():
#                         print("attr", attr)
#                         if attr[0] is attr:
#                             # m3Show.imshow(attr[1],"asdf")
#                             os.makedirs(outputFolder, exist_ok=True)
#                             path = outputFolder  + os.path.basename(photo.path) + "_" + str(eyeCount) + ".jpeg"
#                             print("path", path)
#                             cv2.imwrite(path, attr[1])
#                             eyeCount += 1
