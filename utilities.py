import cv2 as cv
import numpy as np

def StackImages(imgArray,scale,lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        #print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d])*13+27,30+eachImgHeight*d),(255,255,255),cv.FILLED)
                cv.putText(ver,lables[d],(eachImgWidth*c+10,eachImgHeight*d+20),cv.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

def RectContor(contours):
    rectCon = []
    for i in contours:
        area = cv.contourArea(i)
        #print(area)
        if area>50:
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02 * peri,True)
            #print("cornor Points", len(approx))
            if len(approx) == 4:
                rectCon.append(i)

    rectCon = sorted(rectCon, key = cv.contourArea, reverse=True)

    return rectCon

def getCornorPoints(contour):
    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.02 * peri,True)
    return approx

def reorder(mypoints):
    mypoints = mypoints.reshape((4,2))

    mypointsnew = np.zeros((4,1,2),np.int32)

    add = mypoints.sum(1)

    #print(mypoints)
    #print(add)

    mypointsnew[0] = mypoints[np.argmin(add)]   #[0, 0]
    mypointsnew[3] = mypoints[np.argmax(add)]   #[ w, h]

    diff = np.diff(mypoints, axis=1)
    mypointsnew[1] = mypoints[np.argmin(diff)] # [w, 0]
    mypointsnew[2] = mypoints[np.argmax(diff)]  # [ 0,h]
    #print(diff)
    return mypointsnew



def SplitAnswers(img):
    rows = np.vsplit(img,10)
    #cv.imshow("split", rows[0])
    boxes = []
    for r in rows:
        cols = np.hsplit(r,4)
        for box in cols:
            boxes.append(box)
            #cv.imshow("split", box)

    return boxes


def splitRoll(img):
    cols = np.hsplit(img,10)
    #cv.imshow("split", cols[4])

    boxes = []
    for c in cols:
        rows = np.vsplit(c,10)
        for box in rows:
            boxes.append(box)

    return boxes

def splittestid(img):
    cols = np.hsplit(img,5)
    #cv.imshow("split", cols[4])

    boxes = []
    for c in cols:
        rows = np.vsplit(c,10)
        for box in rows:
            boxes.append(box)

    return boxes
