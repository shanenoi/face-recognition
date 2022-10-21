from PIL import Image, ImageTk
from os import path
from tkinter import Tk, Label, Menu, filedialog
import copy
import cv2
import face_recognition as fr
import json
import numpy as np
import pickle


red = (255, 0, 0)
sample = 'sample'


def noise_reducing(img):
    outp_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    outp_img = cv2.medianBlur(outp_img, 3)
    outp_img = cv2.cvtColor(outp_img, cv2.COLOR_HSV2BGR)
    outp_img = cv2.fastNlMeansDenoisingColored(outp_img, None, 7, 7, 7, 21)
    outp_img = cv2.cvtColor(outp_img, cv2.COLOR_BGR2RGB)
    return outp_img


def edge_detecting(img):
    out_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(out_img, 45, 140)


def face_detecting(img):
    detected_locs = fr.face_locations(img)

    out_img = copy.deepcopy(img)
    for (top, right, bottom, left) in detected_locs:
        cv2.rectangle(out_img, (left, top), (right, bottom), red, 2)

    return out_img


def face_recogniting(img):
    face_names = []
    detected_locs = fr.face_locations(img)
    detected_faces = fr.face_encodings(img, detected_locs)

    for f_encoded in detected_faces:
        average_face_distances = []
        for sample_face in sample_faces:
            average_face_distances.append(
                np.round(np.average(
                    fr.face_distance(sample_face, f_encoded)
                ), 1)
            )

        matches = list(np.array(average_face_distances) <= 0.4)
        name = ""
        if True in matches:
            first_idx = matches.index(True)
            name = sample_names[first_idx]
        face_names.append(name)
    out_img = copy.deepcopy(img)

    for (top, right, bottom, left), name in zip(detected_locs, face_names):
        cv2.rectangle(out_img, (left, top), (right, bottom), red, 2)
        cv2.putText(out_img, name, (left, bottom + 30), None, 1.0, red, 2)

    return out_img


def save_var(var, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(var, file)


def load_var(file_name):
    var = None
    with open(file_name, 'rb') as file:
        myvar = pickle.load(file)
        var = myvar
    return var


def load_samples():
    global sample_faces, sample_names
    paths = None

    with open(path.join(sample, 'sample.json')) as jsonfile:
        paths = json.load(jsonfile)

    for name in paths:
        sample_faces_per_one = []
        for img_path in paths[name]:
            sample_img = fr.load_image_file(path.join(sample, img_path))
            sample_faces_per_one.append(fr.face_encodings(sample_img)[0])
        sample_faces.append(sample_faces_per_one)
        sample_names.append(name)

    save_var(sample_faces, sample_faces_data)
    save_var(sample_names, sample_names_data)


def load_samples_data():
    global sample_faces, sample_names
    sample_faces = load_var(sample_faces_data)
    sample_names = load_var(sample_names_data)


sample_faces = []
sample_names = []

sample_faces_data = "sample_faces.data"
sample_names_data = "sample_names.data"

# load_samples()
load_samples_data()

image_name = 'sample/intro.png'
image = None
image_rgb = None
image_gray = None

image_noise_reducing = None
image_edge_detecting = None
image_face_detecting = None
image_face_recogniting = None


def load_image_file():
    global image, image_rgb, image_gray

    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


load_image_file()


def reload_processing_data():
    global image_noise_reducing, image_edge_detecting, \
           image_face_detecting,  image_face_recogniting

    load_image_file()

    image_noise_reducing = noise_reducing(image_rgb)
    image_edge_detecting = edge_detecting(image_noise_reducing)
    image_face_detecting = face_detecting(image_noise_reducing)
    image_face_recogniting = face_recogniting(image_noise_reducing)


def resize_image(img):
    width, height = img.size
    return img.resize([int(600 * width / height), 600])


root_window = Tk()

root_img = Image.fromarray(image_rgb)
imgtk = ImageTk.PhotoImage(image=resize_image(root_img))

panel = Label(root_window, image=imgtk)
panel.pack(side="bottom", fill="both", expand="yes")


def tk_open_file():
    global image_name
    image_name = filedialog.askopenfilename()
    reload_processing_data()

    img = Image.fromarray(image_rgb)
    imgtk = ImageTk.PhotoImage(image=resize_image(img))

    panel.configure(image=imgtk)
    panel.image = imgtk


def original():
    img = Image.fromarray(image_rgb)
    imgtk = ImageTk.PhotoImage(image=resize_image(img))

    panel.configure(image=imgtk)
    panel.image = imgtk


def noiseReduce():
    img = Image.fromarray(image_noise_reducing)
    imgtk = ImageTk.PhotoImage(image=resize_image(img))

    panel.configure(image=imgtk)
    panel.image = imgtk


def edgeDetect():
    img = Image.fromarray(image_edge_detecting)
    imgtk = ImageTk.PhotoImage(image=resize_image(img))

    panel.configure(image=imgtk)
    panel.image = imgtk


def faceDetecting():
    img = Image.fromarray(image_face_detecting)
    imgtk = ImageTk.PhotoImage(image=resize_image(img))

    panel.configure(image=imgtk)
    panel.image = imgtk


def faceRecognize():
    img = Image.fromarray(image_face_recogniting)
    imgtk = ImageTk.PhotoImage(image=resize_image(img))

    panel.configure(image=imgtk)
    panel.image = imgtk


menu = Menu(root_window)
filemn = Menu(menu, tearoff=0)
filemn.add_command(label="Open Image", command=tk_open_file)
filemn.add_separator()
filemn.add_command(label="Exit", command=root_window.quit)
menu.add_cascade(label="File", menu=filemn)

image_processing = Menu(menu, tearoff=0)
image_processing.add_command(label="Original", command=original)
image_processing.add_command(label="Noise Reduce", command=noiseReduce)
image_processing.add_command(label="Edge Detect", command=edgeDetect)
image_processing.add_command(label="Face Detect", command=faceDetecting)
image_processing.add_command(label="Face Recognize", command=faceRecognize)
menu.add_cascade(label="Actions", menu=image_processing)

root_window.config(menu=menu)
root_window.mainloop()
