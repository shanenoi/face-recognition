import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter

def noise_reduce():
    figure_size = 5
    ##new_image = cv2.GaussianBlur(image, (figure_size, figure_size),0)
    new_image = cv2.medianBlur(image, figure_size)
    return new_image

def detect_edge():
    new_image = cv2.cvtColor(image_less_noise,cv2.COLOR_BGR2GRAY)
    new_image = cv2.Canny(new_image,40,140)
    return new_image

def detect_face():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image_less_noise,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)
    new_image = noise_reduce()
    for (x, y, w, h) in faces:
        cv2.rectangle(new_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return new_image
    

image = cv2.imread('dog.jpg')
##image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

image_less_noise = noise_reduce()
image_detect_edge = detect_edge()
image_detect_face = detect_face()

plt.figure(figsize=(11,6))
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR)),plt.title('original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(cv2.cvtColor(image_detect_face, cv2.COLOR_RGB2BGR)),plt.title('face detect')
plt.xticks([]), plt.yticks([])
plt.show()



