import cv2 as cv
import numpy as np
import utilities

################################################################

path = "Photos/FinalTest.jpg"
width = 700
height = 700
Questions = 30
Choices = 4

Answers = [0, 1, 0, 2, 2, 3, 0, 1, 2, 3, 1, 0, 0, 2, 0, 3, 0, 1, 2, 3, 0, 1, 0, 2, 2, 3, 2, 0, 2, 3]
#print(len(Answers))
####################################################################

img = cv.imread(path)

def Scanner(img):
    img = cv.resize(img, (width, height))
    imgcontors = img.copy()
    imgbigestcontors = img.copy()
    imggray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgblur = cv.GaussianBlur(imggray, (5,5),1)
    imgcanny =cv.Canny(imgblur, 10,50)


    ### finding contours
    contours, hierachy = cv.findContours(imgcanny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.drawContours(imgcontors, contours, -1 ,(0,255,0),10)
    ## find Rectangles
    rectCon = utilities.RectContor(contours)
    RollNo = utilities.getCornorPoints(rectCon[0])
    testid = utilities.getCornorPoints(rectCon[5])
    #print(RollNo.shape)
    Question1 = utilities.getCornorPoints(rectCon[2])
    Question2 = utilities.getCornorPoints(rectCon[3])
    Question3 = utilities.getCornorPoints(rectCon[4])


    #print(largestContor)

    if RollNo.size != 0 and Question1.size !=0 and Question2.size != 0:
        cv.drawContours(imgbigestcontors, RollNo, -1, (255,0,0),20)
        cv.drawContours(imgbigestcontors, testid, -1, (255,255,0),20)
        cv.drawContours(imgbigestcontors,Question1,-1,(0,255,0),20)
        cv.drawContours(imgbigestcontors,Question2,-1,(0,0,255),20)
        cv.drawContours(imgbigestcontors,Question3,-1,(255,0,255),20)

        RollNo = utilities.reorder(RollNo)
        testid = utilities.reorder(testid)
        Question1 = utilities.reorder(Question1)
        Question2 = utilities.reorder(Question2)
        Question3 = utilities.reorder(Question3)

        pt1 = np.float32(RollNo)
        pt2 = np.float32([[0,0],[width,0],[0,height],[width, height]])
        matrix = cv.getPerspectiveTransform(pt1,pt2)
        RollNoWrapped = cv.warpPerspective(img,matrix,(width,height))

        pt9 = np.float32(testid)
        pt10 = np.float32([[0,0],[width,0],[0,height],[width, height]])
        matrix4 = cv.getPerspectiveTransform(pt9,pt10)
        testidWrapped = cv.warpPerspective(img,matrix4,(width,height))

        pt3 = np.float32(Question1)
        pt4 = np.float32([[0,0],[width,0],[0,height],[width, height]])
        matrix1 = cv.getPerspectiveTransform(pt3,pt4)
        Question1Wraped = cv.warpPerspective(img,matrix1,(width,height))

        pt5 = np.float32(Question2)
        pt6 = np.float32([[0,0],[width,0],[0,height],[width, height]])
        matrix2 = cv.getPerspectiveTransform(pt5,pt6)
        Question2Wraped = cv.warpPerspective(img,matrix2,(width,height))

        pt7 = np.float32(Question3)
        pt8 = np.float32([[0,0],[width,0],[0,height],[width, height]])
        matrix3 = cv.getPerspectiveTransform(pt7,pt8)
        Question3Wraped = cv.warpPerspective(img,matrix3,(width,height))




        #appliing Threshold For Detection
        RollNoWrappedGray = cv.cvtColor(RollNoWrapped, cv.COLOR_BGR2GRAY)
        RollNoTresh = cv.threshold(RollNoWrappedGray, 170, 255, cv.THRESH_BINARY_INV)[1]

        testidWrappedGray = cv.cvtColor(testidWrapped, cv.COLOR_BGR2GRAY)
        testidTresh = cv.threshold(testidWrappedGray, 170, 255, cv.THRESH_BINARY_INV)[1]

        Question1WrapedGray = cv.cvtColor(Question1Wraped, cv.COLOR_BGR2GRAY)
        Question1Tresh = cv.threshold(Question1WrapedGray, 170, 255, cv.THRESH_BINARY_INV)[1]

        Question2WrapedGray = cv.cvtColor(Question2Wraped, cv.COLOR_BGR2GRAY)
        Question2Tresh = cv.threshold(Question2WrapedGray, 170, 255, cv.THRESH_BINARY_INV)[1]

        Question3WrapedGray = cv.cvtColor(Question3Wraped, cv.COLOR_BGR2GRAY)
        Question3Tresh = cv.threshold(Question3WrapedGray, 170, 255, cv.THRESH_BINARY_INV)[1]

        box1 = utilities.SplitAnswers(Question1Tresh)
        box2 = utilities.SplitAnswers(Question2Tresh)
        box3 = utilities.SplitAnswers(Question3Tresh)

        boxes = box1+box2+box3

        #print(len(boxes))
        #cv.imshow("test", boxes[1])
        #print(cv.countNonZero(boxes[1]))
        #print(cv.countNonZero(boxes[2]))


        # Getting NonZero Pixal Values
        myPixalValue = np.zeros((Questions, Choices))
        countC = 0
        countR = 0

        for image in boxes:
            totalpixals = cv.countNonZero(image)
            myPixalValue[countR][countC] =  totalpixals
            countC += 1
            if( countC == Choices):
                countR += 1
                countC = 0

        np.set_printoptions(suppress=True)
        #print(myPixalValue)

        ################   Finding Index Values of Markings
        myIndex = []
        for x in range(0,Questions):
            arr = myPixalValue[x]
            myIndexVal = np.where(arr == np.amax(arr))
            #print(myIndexVal[0])
            myIndex.append(myIndexVal[0][0])
        #print(myIndex)


        # Grading
        grading = []
        for x in range(0,Questions):
            if Answers[x] == myIndex[x]  and myPixalValue[x][myIndex[x]] >= 2000:
                cnt = 0
                for i in range(0,4):
                    if(myPixalValue[x][i] >= 2000):
                        cnt += 1

                if(cnt > 1):
                    grading.append(0)
                else:
                    grading.append(1)
            else:
                grading.append(0)

        #print(Answers)
        #print("")
        #print(grading)
        Marks_obtained = sum(grading)




        ### checking for Roll no
        RollBoxes = utilities.splitRoll(RollNoTresh)
        myPixalValue_roll = np.zeros((10, 10))
        countC = 0
        countR = 0

        for image in RollBoxes:
            totalpixals = cv.countNonZero(image)
            myPixalValue_roll[countR][countC] =  totalpixals
            countC += 1
            if( countC == 10):
                countR += 1
                countC = 0

        np.set_printoptions(suppress=True)
        #print(myPixalValue_roll)

        ################   Finding Index Values of Markings
        myIndex_roll = []
        for x in range(0,10):
            arr = myPixalValue_roll[x]
            myIndexVal_roll = np.where(arr == np.amax(arr))
            #print(myIndexVal[0])
            myIndex_roll.append(myIndexVal_roll[0][0])
        #print(myIndex_roll)
        Scanned_Roll = ''.join(map(str, myIndex_roll))





        ### checking for testID
        testid_box = utilities.splittestid(testidTresh)
        #print(testidTresh)
        myPixalValue_test = np.zeros((5, 10))
        countC = 0
        countR = 0

        for image in testid_box:
            totalpixals = cv.countNonZero(image)
            myPixalValue_test[countR][countC] =  totalpixals
            countC += 1
            if( countC == 10):
                countR += 1
                countC = 0

        np.set_printoptions(suppress=True)
        #print(myPixalValue_test)

        ################   Finding Index Values of Markings
        myIndex_test = []
        for x in range(0,5):
            arr = myPixalValue_test[x]
            myIndexVal_test = np.where(arr == np.amax(arr))
            #print(myIndexVal_test[0])
            myIndex_test.append(myIndexVal_test[0][0])
        #print(type(myIndex_test))
        Scanned_testID = ''.join(map(str, myIndex_test))






    imageBlank =np.zeros_like(img)

    imageAray = ([img, imggray, imgblur, imgcanny],
                 [imgcontors, imgbigestcontors, Question1Wraped, Question1Tresh])
    imgStacked = utilities.StackImages(imageAray, 0.5)

    #cv.imshow("Stacked Images", imgStacked)
    #cv.waitKey(0)

    return Scanned_Roll, Scanned_testID, Marks_obtained

rollno, testid, Marks =  Scanner(img)

