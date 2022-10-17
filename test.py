import numpy as np
import cv2
import os
import face_recognition
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter


hello = 'sample//Bao.jpg'

name = hello.split(os.path.sep)[-2]

print(name)