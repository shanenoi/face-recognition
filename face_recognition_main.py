from PIL import Image, ImageTk
from imutils import paths
from matplotlib import pyplot as plt
from tkinter import Tk, Label, Menu, Toplevel, filedialog
import copy
import cv2
import face_recognition
import numpy as np
import os
import pickle

def noise_reduce(img):
    ##figure_size = 5
    ##new_image = cv2.GaussianBlur(image, (figure_size, figure_size),0)
    ##new_image = cv2.medianBlur(img, figure_size)
    new_image = cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)
    return new_image

def detect_edge(img):
    new_image = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    new_image = cv2.Canny(new_image,40,140)
    return new_image

def detect_face(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image_gray, 1.1, 5)
    new_image = copy.deepcopy(image_less_noise)
    for (x, y, w, h) in faces:
        cv2.rectangle(new_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return new_image

def recognition_face(img):
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = ""
        # if True in matches:
        #    first_match_index = matches.index(True)
        #    name = known_face_names[first_match_index]
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index] and face_distances[best_match_index] < 0.4:
            name = known_face_names[best_match_index]
            print(name)
            print(face_distances)
        face_names.append(name)
    new_image = copy.deepcopy(img)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(new_image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(new_image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(new_image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    return new_image

ky_imgae = face_recognition.load_image_file("sample/Ky.jpg")
ky_face_encoding = face_recognition.face_encodings(ky_imgae)[0]

van_imgae = face_recognition.load_image_file("sample/Van.jpg")
van_face_encoding = face_recognition.face_encodings(van_imgae)[0]

nv_imgae = face_recognition.load_image_file("sample/NV.jpg")
nv_face_encoding = face_recognition.face_encodings(nv_imgae)[0]

dv_imgae = face_recognition.load_image_file("sample/DV.jpg")
dv_face_encoding = face_recognition.face_encodings(dv_imgae)[0]

bao_imgae = face_recognition.load_image_file("sample/Bao.jpg")
bao_face_encoding = face_recognition.face_encodings(bao_imgae)[0]

known_face_encodings = [
    nv_face_encoding,
    dv_face_encoding,
    bao_face_encoding,
    ky_face_encoding,
    van_face_encoding
]


known_face_names = [
    "N.Vu",
    "D.Vu",
    "Bao",
    "Ky",
    "Van"
]


image_name = 'group2.jpg'
image = None
image_rgb = None
image_gray = None

image_less_noise = None
image_detect_edge = None
image_detect_face = None
image_recognize_face = None


def reload_processing_data():
    global image, image_rgb, image_gray, \
            image_less_noise, image_detect_edge, \
            image_detect_face, image_recognize_face

    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_less_noise = noise_reduce(image_rgb)
    image_detect_edge = detect_edge(image_less_noise)
    image_detect_face = detect_face(image_rgb)
    image_recognize_face = recognition_face(image_less_noise)


reload_processing_data()

# plt.subplot(121),plt.imshow(image_rgb)
# plt.subplot(121),plt.imshow(image_recognize_face)
# plt.show()

# cv2.imshow("original", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
# cv2.imshow("gray", image_gray)
# cv2.imshow("noise reduce", cv2.cvtColor(image_less_noise, cv2.COLOR_RGB2BGR))
# cv2.imshow("edge detect", image_detect_edge)
# cv2.imshow("face detect", cv2.cvtColor(image_detect_face, cv2.COLOR_RGB2BGR))
# cv2.imshow("face recognize", cv2.cvtColor(image_recognize_face, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)

root = Tk()

im = Image.fromarray(image)
imgtk = ImageTk.PhotoImage(image=im)

panel = Label(root, image=imgtk)
panel.pack(side="bottom", fill="both", expand="yes")


def callback():
    global image_name
    image_name = filedialog.askopenfilename()
    reload_processing_data()

    im = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=im)

    panel.configure(image=imgtk)
    panel.image = imgtk


def noiseReduce():
    im = Image.fromarray(image_less_noise)
    imgtk = ImageTk.PhotoImage(image=im)

    panel.configure(image=imgtk)
    panel.image = imgtk


def edgeDetect():
    im = Image.fromarray(image_detect_edge)
    imgtk = ImageTk.PhotoImage(image=im)

    panel.configure(image=imgtk)
    panel.image = imgtk


def faceRecognize():
    im = Image.fromarray(image_recognize_face)
    imgtk = ImageTk.PhotoImage(image=im)

    panel.configure(image=imgtk)
    panel.image = imgtk


menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Open Image", command=callback)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=filemenu)

actions = Menu(menubar, tearoff=0)
actions.add_command(label="Noise Reduce", command=noiseReduce)
actions.add_command(label="Edge Detect", command=edgeDetect)
actions.add_command(label="Face Recognize", command=faceRecognize)
menubar.add_cascade(label="Actions", menu=actions)

root.config(menu=menubar)
root.mainloop()
