import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm_notebook as tqdm
from matplotlib import pyplot as plt
from PIL import Image
import torch
import time

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
from facenet_pytorch import MTCNN
detector = MTCNN(device=device)

def detect_facenet_pytorch(detector, images, batch_size=20):
    faces = []
    for lb in np.arange(0, len(images), batch_size):
        imgs_pil = [Image.fromarray(image) for image in images[lb:lb+batch_size]]
        faces.extend(detector(imgs_pil))
    return faces

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

batch_size = 20
while cv2.waitKey(33) != ord('q'):
    ret, frame = capture.read()
    faces = detect_facenet_pytorch(detector, frame, batch_size)
    cv2.imshow("FACE", faces)

capture.release()
cv2.destroyAllWindows()
