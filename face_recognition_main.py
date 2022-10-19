from PIL import Image, ImageTk
from tkinter import Tk, Label, Menu, filedialog
import copy
import cv2
import face_recognition as fr
import json
from os import path


yellow = (0, 255, 0)
sample = 'sample'


def noise_reducing(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)


def edge_detecting(img):
    out_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(out_img, 40, 140)


def face_recogniting(img):
    face_names = []
    detected_locs = fr.face_locations(img)
    detected_faces = fr.face_encodings(img, detected_locs)

    for f_encoded in detected_faces:
        matches = fr.compare_faces(sample_faces, f_encoded, 0.4)
        name = ""
        if True in matches:
            first_idx = matches.index(True)
            name = sample_names[first_idx]
        face_names.append(name)
    out_img = copy.deepcopy(img)

    for (top, right, bottom, left), name in zip(detected_locs, face_names):
        cv2.rectangle(out_img, (left, top), (right, bottom), yellow, 2)
        cv2.putText(out_img, name, (left, bottom + 28), None, 1.0, yellow, 2)

    return out_img


def load_samples():
    global sample_faces, sample_names
    paths = None

    with open(path.join(sample, 'sample.json')) as jsonfile:
        paths = json.load(jsonfile)

    for name in paths:
        sample_img = fr.load_image_file(path.join(sample, paths[name]))
        sample_faces.append(fr.face_encodings(sample_img)[0])
        sample_names.append(name)


def load_image_file():
    global image, image_rgb, image_gray

    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# image_name = 'intro.jpg'
image_name = '/tmp/abc.png'
image = None
image_rgb = None
image_gray = None

image_less_noise = None
image_detect_edge = None
image_recognize_face = None

sample_faces = []
sample_names = []

load_samples()
load_image_file()


def reload_processing_data():
    global image_less_noise, image_detect_edge, image_recognize_face

    load_image_file()

    image_less_noise = noise_reducing(image_rgb)
    image_detect_edge = edge_detecting(image_less_noise)
    image_recognize_face = face_recogniting(image_less_noise)


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
    img = Image.fromarray(image_less_noise)
    imgtk = ImageTk.PhotoImage(image=resize_image(img))

    panel.configure(image=imgtk)
    panel.image = imgtk


def edgeDetect():
    img = Image.fromarray(image_detect_edge)
    imgtk = ImageTk.PhotoImage(image=resize_image(img))

    panel.configure(image=imgtk)
    panel.image = imgtk


def faceRecognize():
    img = Image.fromarray(image_recognize_face)
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
image_processing.add_command(label="Face Recognize", command=faceRecognize)
menu.add_cascade(label="Actions", menu=image_processing)

root_window.config(menu=menu)
root_window.mainloop()
