import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from tqdm import tqdm_notebook as tqdm
import os

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(device=device)

capture = cv2.VideoCapture(0)

while cv2.waitKey(33) != ord('q'):
    ret, frame = capture.read()
    faces, _ = mtcnn.detect(frame)
    #print(faces)
    try:    
        for face in faces:
            face = np.trunc(face)
            cv2.rectangle(frame, (face[0],face[1]), (face[2], face[3]), (255,0,0), 3)
        
    except:
        pass
    cv2.imshow("FACE", frame)
capture.release()
cv2.destroyAllWindows()
