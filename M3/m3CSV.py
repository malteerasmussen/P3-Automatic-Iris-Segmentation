# TP, FN, FP

import csv
def makeCSV(photoArray, fileToSave):
    with open(fileToSave, 'w', newline='') as csvfile:
        fieldnames = ['image_name','eye','noCircles',"circles", 'TP', 'TN', 'FP', 'FN', "TPFN", "rTP", "rTN", "rFP", "rFN", "rHandAccum", "rAutoAccum", "imHeight", "imWidth" ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for photo in photoArray:
            for face in photo.faces:
                eyeCount = 0
                for eye in face.eyes:
                    if eye.noCircles:
                        numOfCircles = 0
                    else:
                        numOfCircles = len(eye.circles)
                    writer.writerow({'image_name': photo.path, 'eye': eyeCount, 'noCircles': eye.noCircles, 'circles': numOfCircles, 'TP': eye.TP,  'TN': eye.TN, 'FP': eye.FP, 'FN': eye.FN, "TPFN": eye.TPFN, "rTP": eye.rTP, "rTN": eye.rTN, "rFP": eye.rFP, "rFN": eye.rFN, "rHandAccum": eye.rHandAccum, "rAutoAccum": eye.rAutoAccum, "imHeight": eye.image.shape[0], "imWidth": eye.image.shape[1]   })
                    eyeCount += 1
