import streamlit as st
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import math

bytes_data = []

uploaded_files = st.file_uploader("Choose a Image file", type=['png', 'jpg'], accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    # st.write(bytes_data)

# st.image(bytes_data)

model2 = YOLO('models/ANPR_model2.1.pt')
model3 = YOLO('models/ANPR_model3.1.pt')
model4 = YOLO('models/ANPR_model4.1.pt')

# Open the video file
# video_path = "videos/video2.mp4"
# cap = cv2.VideoCapture(video_path)

if len(bytes_data) > 0:
    nparr = np.fromstring(bytes_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # img = cv2.imread('images/555.png')
    # img = bytes_data

    track_history = defaultdict(lambda: [])
    checkIdRev = None

    results1 = model2.track(img, persist=True)

    boxes = results1[0].boxes.xywh.cpu()
    confs = results1[0].boxes.conf
    annotated_frame = results1[0].plot()
    cls = int(results1[0].boxes.cls)


    ######################
    # Filter

    def noise_removal(image):
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.medianBlur(image, 3)
        return image


    def thin_font(image):
        image = cv2.bitwise_not(image)
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.bitwise_not(image)
        return image


    def thick_font(image):
        image = cv2.bitwise_not(image)
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        image = cv2.bitwise_not(image)
        return image


    def remove_borders(image):
        contours, heiarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
        cnt = cntsSorted[-1]
        x, y, w, h = cv2.boundingRect(cnt)
        crop = image[y:y + h, x:x + w]
        return crop


    #################################################

    def model1Cls():
        if cls == 0:
            print('single row number plate')
            for box in boxes:
                x, y, w, h = box
                x1, y1 = x - (w / 2), y - (h / 2)
                x1, y1, w, h = int(x1), int(y1), int(w), int(h)
                cropped = annotated_frame[y1:y1 + h, x1:x1 + w]
                crpImgHeight, crpImgWidth = cropped.shape[0], cropped.shape[1]
                cropped = cv2.resize(cropped, (5 * crpImgWidth, 5 * crpImgHeight))  # cropped
                # cropped = cv2.medianBlur(cropped, 5)
                # cropped = cv2.GaussianBlur(cropped, (5, 5), 0)

            return 0, cropped

        elif cls == 1:
            print('double raw number plate')
            for box in boxes:
                x, y, w, h = box
                x1, y1 = x - (w / 2), y - (h / 2)
                x1, y1, w, h = int(x1), int(y1), int(w), int(h)
                cropped = annotated_frame[y1:y1 + h, x1:x1 + w]
                crpImgHeight, crpImgWidth = cropped.shape[0], cropped.shape[1]
                cropped = cv2.resize(cropped, (2 * crpImgWidth, 2 * crpImgHeight))
                # cropped = cv2.medianBlur(cropped, 5)
                # cropped = cv2.GaussianBlur(cropped, (5,5), 0)

            return 1, cropped


    def modifyImg(img, thresh=130, maxval=220):
        blank_image = np.ones((200, 500, 3), np.uint8) * 255
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur1 = cv2.medianBlur(img1, 5)
        blur2 = cv2.GaussianBlur(gray, (9, 9), 0)
        # inverted_image = cv2.bitwise_not(blur2)
        thresh, im_bw = cv2.threshold(blur2, thresh, maxval, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(image=im_bw, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(image=blank_image, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=3,
                         lineType=cv2.LINE_AA)

        no_noise = noise_removal(im_bw)
        eroded_image = thin_font(no_noise)

        # cv2.imshow("image", im_bw)
        # cv2.waitKey(0)

        return eroded_image


    def readText(image, cls):

        if cls == 0:
            reader = easyocr.Reader(['en'])
            result = reader.readtext(image)

        elif cls == 1:
            reader = easyocr.Reader(['en'])
            result = reader.readtext(image)

        return result


    def numbersCheck(input_string):
        return all(char.isdigit() for char in input_string)


    text1 = ''
    number1 = ''
    text2 = ''
    number2 = ''

    plateType, img1 = model1Cls()

    # cv2.imshow("Cropped Number Plate", img1)
    # cv2.waitKey(0)

    if plateType == 0:
        results2 = model3.track(img1, persist=True)
        boxes = results2[0].boxes.xywh.cpu()
        confs = results2[0].boxes.conf
        annotated_frame1 = results2[0].plot()
        clsses = results2[0].boxes.cls

        st.image(annotated_frame1)

        for box, cls in zip(boxes, clsses):
            print(cls)
            x, y, w, h = box
            x1, y1 = x - (w / 2), y - (h / 2)
            x1, y1, w, h = int(x1), int(y1), int(w), int(h)
            cropped = img1[y1:y1 + h, x1:x1 + w]

            inc = 0
            if int(cls) == 0.0:
                modImg = modifyImg(cropped)
                # cv2.imshow('Filtered Letters', modImg)
                # cv2.waitKey(0)
                text1 = readText(modImg, cls)
                if len(text1) > 0:
                    if len(text1[0][1]) == 2 or len(text1[0][1]) == 3:
                        print("Letters Catched")
                    # else:
                    #     modImg = modifyImg(cropped, 150, 230)
                    #     text1 = readText(modImg, cls)

            if int(cls) == 1.0:
                modImg = modifyImg(cropped)
                # cv2.imshow('Filtered Number', modImg)
                # cv2.waitKey(0)
                number1 = readText(modImg, cls)
                if len(number1) > 0:
                    if len(number1[0][1]) == 4 and numbersCheck(number1[0][1]):
                        print("Numbers Catched")
                    # else:
                    #     modImg = modifyImg(cropped, 150, 230)
                    #     number1 = readText(modImg, cls)

        text = " "
        number = " "
        if (len(text1) > 0):
            text = text1[0][1]

        if (len(number1) > 0):
            number = number1[0][1]

        if len(text1) == 0 and len(number1) == 0:
            st.write("Cant Read Number Plate")

        else:
            st.write("Plate Number : " + text + " " + number)

    elif plateType == 1:
        results3 = model4.track(img1, persist=True)
        boxes = results3[0].boxes.xywh.cpu()
        confs = results3[0].boxes.conf
        annotated_frame1 = results3[0].plot()
        clsses = results3[0].boxes.cls

        st.image(annotated_frame1)

        for box, cls in zip(boxes, clsses):
            print(cls)
            x, y, w, h = box
            x1, y1 = x - (w / 2), y - (h / 2)
            x1, y1, w, h = int(x1), int(y1), int(w), int(h)
            cropped = img1[y1:y1 + h, x1:x1 + w]

            if int(cls) == 0.0:
                modImg = modifyImg(cropped)
                # cv2.imshow('Filtered Letters', modImg)
                # cv2.waitKey(0)
                text2 = readText(modImg, cls)
                if len(text2) > 0:
                    if len(text2[0][1]) == 2 or len(text2[0][1]) == 3:
                        print("Letters Catched")
                    # else:
                    #     modImg = modifyImg(cropped, 150, 230)
                    #     text2 = readText(modImg, cls)

            if int(cls) == 1.0:
                modImg = modifyImg(cropped)
                # cv2.imshow('Filtered Numbers', modImg)
                # cv2.waitKey(0)
                number2 = readText(modImg, cls)
                if len(number2) > 0:
                    if len(number2[0][1]) == 4 and numbersCheck(number2[0][1]):
                        print("Numbers Catched")
                    # else:
                    #     modImg = modifyImg(cropped, 150, 230)
                    #     number2 = readText(modImg, cls)

        text = " "
        number = " "
        if (len(text2) > 0):
            text = text2[0][1]

        if (len(number2) > 0):
            number = number2[0][1]

        if len(text2) == 0 and len(number2) == 0:
            st.write("Cant Read Number Plate")

        else:
            st.write("Plate Number : " + text + " " + number)
