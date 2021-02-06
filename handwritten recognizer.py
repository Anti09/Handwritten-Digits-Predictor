import tkinter as tk  # a standard GUI library for python tkinter
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import cv2
import joblib

win = tk.Tk()

# Main Window just a window will appear in output
# win.mainloop()
w, h = 500, 500
fontDesign = 'Courier 20 bold'
fontLabelS = 'Courier 24 bold'
cnt = 0

model = joblib.load('KNN-HANDWRITTEN-DIGITS.sav')


def eventFunction(event):
    x = event.x
    y = event.y

    x1 = x - 40  # 30,20,10.... just for the thickness
    y1 = y - 40
    x2 = x1 + 40
    y2 = y1 + 40

    canvas.create_oval((x1, y1, x2, y2), fill='black')  # from tkinter lib
    imgDraw.ellipse((x1, y1, x2, y2), fill='white')  # from pillow library


def save():
    global cnt
    imgArray = np.array(img)  # converting image obj into an array
    imgArray = cv2.resize(imgArray, (8, 8))

    cv2.imwrite(str(cnt) + '.jpg', imgArray)  # file name 'data/0.jpg
    cnt = cnt + 1
    print("Image Saved!")


def clear():
    global img, imgDraw

    canvas.delete('all')
    img = Image.new('RGB', (w, h), (0, 0, 0))
    imgDraw = ImageDraw.Draw(img)


def predict():
    '''pre_process the img'''
    imgArray = np.array(img)
    # RGB to GRAYSCALE conversion
    imgArray = cv2.cvtColor(imgArray, cv2.COLOR_RGB2GRAY)
    imgArray = cv2.resize(imgArray, (8, 8))

    # flattening the image
    imgArray = np.reshape(imgArray, (1, 64))
    print(imgArray)

    # scaling to 0-16
    imgArray = (imgArray / 255.0) * 15.0
    res = model.predict(imgArray)
    labelStatus.config(text="PREDICTED DIGIT:-" + str(res))


canvas = tk.Canvas(win, width=w, height=h)  # canvas is from tkinter library
canvas.grid(row=0, column=0, columnspan=4)

# SAVE Button
buttonSave = tk.Button(win, text='Save', bg='MediumSeaGreen', fg='white', font=fontDesign, command=save)
buttonSave.grid(row=1, column=0)

# PREDICTION button
buttonPredict = tk.Button(win, text='Predict', bg='Tomato', fg='white', font=fontDesign, command=predict)
buttonPredict.grid(row=1, column=1)

# CLEAR Button
buttonClear = tk.Button(win, text='Clear', bg='Sky blue', fg='white', font=fontDesign, command=clear)
buttonClear.grid(row=1, column=2)

# EXIT Button
buttonExit = tk.Button(win, text='Exit', bg='light pink', fg='white', font=fontDesign, command=win.destroy)
buttonExit.grid(row=1, column=3)

labelStatus = tk.Label(win, text='PREDICT : NONE', bg='white', fg='black', font=fontLabelS)
labelStatus.grid(row=2, column=0, columnspan=4)

canvas.bind('<B1-Motion>', eventFunction)
# B1 is left click
# B2 is Scroll
# B3 is Right click

img = Image.new('RGB', (w, h), (0, 0, 0))  # image obj
imgDraw = ImageDraw.Draw(img)  # actual obj to draw

win.mainloop()
