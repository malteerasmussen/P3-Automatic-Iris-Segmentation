import cv2 
import PIL 
import numpy as np
    
def cntImgPxl (inputImg1, inputImg2): 
    

    #INPUTIMAGE 2 IS THE BASECASE - The handmask
    #Important value !!!overlapingWhitePixelPer!!!!

    #inputImg1 = cv2.imread(inputImg1,cv2.IMREAD_GRAYSCALE)
    #inputImg2 = cv2.imread(inputImg2,cv2.IMREAD_GRAYSCALE)
    inputImg1 =cv2.IMREAD_GRAYSCALE(inputImg1)
    inputImg2 =cv2.IMREAD_GRAYSCALE(inputImg2)
    

    ret,binary1 = cv2.threshold(inputImg1,127,255,cv2.THRESH_BINARY)
    ret,binary2 = cv2.threshold(inputImg2,127,255,cv2.THRESH_BINARY)

    rows1,cols1 = inputImg1.shape
    #Pixel-count in inputImg1
    cntPixels1 = rows1*cols1

    count1= 0
        for i in range(rows1):
            for j in range(cols1):
                #Pixel-value counter for image
                count1 += inputImg1[i,j]
        
        
                #Non-black pixels
                cntNotBlack1 = cv2.countNonZero(binary1)
                #Black-pixels
                cntBlack1 = cntPixels1 - cntNotBlack1
                #Black pixel percentage for binary
                blackPixelPer1 = (cntBlack1/cntPixels1)*100
        
                #White-pixels !OBS for non-binanry!
                n_white_pix1 = np.sum(inputImg1 == 255)
                #Percentage of white pixels !OBS for non-binanry!
                whitePixelPer1 = (n_white_pix1/cntPixels1)*100  

    #Normalize image
    # As every pixel which is not black (0), has a value between 1-255 the image will be normalized,
    # Each white pixel will weight 1 while ranging between 1-254 will weight pixel/255 >1 
    # This way every pixel value is taking into consideration
    count1= count1/255
    print("Value of pixels normalized in inputImage 1:",count1)




    rows2,cols2 = inputImg2.shape
    #Pixel-count in inputImg2
    cntPixels2 = rows2*cols2

    count2= 0
    for i in range(rows2):
        for j in range(cols2):

            #Pixel-value counter for image
            count2 += inputImg2[i,j]


            #Non-black pixels
            cntNotBlack2 = cv2.countNonZero(binary2)
            #Black-pixels
            cntBlack2 = cntPixels2 - cntNotBlack2
            #Black pixel percentage for binary
            blackPixelPer2 = (cntBlack2/cntPixels2)*100

            #White-pixels !OBS for non-binanry!
            n_white_pix2 = np.sum(inputImg2 == 255)
            #Percentage of white pixels !OBS for non-binanry!        
            whitePixelPer2 = (n_white_pix2/cntPixels2)*100



    #Normalize image       
    count2= count2/255
    print("Value of pixels normalized in inputImage 2:",count2)

    #Black pixel overlapping percentage between inputImage 1 and 2. 
    blackPixelComparison = (blackPixelPer2/blackPixelPer1)*100
    #White pixel overlapping percentage between inputImage 1 and 2. 
    whitePixelComparison = (whitePixelPer2/whitePixelPer1)*100


    # https://www.studieportalen.dk/kompendier/matematik/formelsamling/procentregning/procent/procentvis-afvigelse
    #GRAYPIXELS + WHITE
    #Procentvis afvigelse af alt ikke sort
    deviatingValuedPixelPer = (abs(count1-count2)/count2)*100
    overlappingValuedPixelPer = 100-deviatingValuedPixelPer

    print ("Percentage deviation of valued pixels from BASECASE",deviatingValuedPixelPer)
    print ("Percentage overlapping of valued pixels from BASECASE",overlappingValuedPixelPer)

    #ONLY WHITE
    #Vægtede andel af hvide pixels 
    whitePixels1 =(n_white_pix1/cntNotBlack1)*100
    whitePixels2 =(n_white_pix2/cntNotBlack2)*100

    #Procentvis afvigelse af vægtede hvide pixels
    deviatingWhitePixelPer = (abs(whitePixels1-whitePixels2)/whitePixels2)*100
    overlapingWhitePixelPer = 100-deviatingWhitePixelPer

    print ("Percentage deviation of white pixels from BASECASE",deviatingWhitePixelPer)
    #FINAL RESULT
    print ("Percentage overlapping of white pixels from BASECASE",overlapingWhitePixelPer)
    print ("Non-black pixels for inputImage 1:",cntNotBlack1)
    print ("Non-black pixels for inputImage 2:",cntNotBlack2)
    print ("Pixels for inputImage 1:",cntPixels1)
    print ("Pixels for inputImage 2:",cntPixels2)
    print ("Black pixels for inputImage 1:",cntBlack1)
    print ("Black pixels for inputImage 2:",cntBlack2)
    print ("Black pixels percentage for inputImage 1:",blackPixelPer1)
    print ("Black pixels percentage for inputImage 2:",blackPixelPer2)
    print ("Percentage of overlapping black pixels in inputImage 1 and 2:",blackPixelComparison)
    print ("White pixels for inputImage 1:",n_white_pix1)
    print ("White pixels for inputImage 2:",n_white_pix2)
    print ("White pixels percentage for inputImage 1:",whitePixelPer1)
    print ("White pixels percentage inputImage 2:",whitePixelPer2)
    print ("Percentage of overlapping white pixels in inputImage 1 and 2:",whitePixelComparison)
    
    return (overlapingWhitePixelPer)






