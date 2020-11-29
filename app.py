import os.path
import numpy as np
import cv2 as cv
import json
from flask import Flask, request, Response, jsonify
import read, utilities


app = Flask(__name__)

@app.route('/')

def upload():
    img = cv.imread("Photos/FinalTest.jpg")

    roll, testid, marks = read.Scanner(img)
    return jsonify(Roll_No = roll,
                   Test_id = testid,
                   Marks = marks)

if __name__ == "__main__":
    app.run(debug=True)

