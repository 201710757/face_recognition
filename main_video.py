import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from tqdm import tqdm_notebook as tqdm
import os
from FastMTCNN import FastMTCNN


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

FastVersion = True 
mtcnn = FastMTCNN(stride=4, resize=1, margin=14, factor=0.6, keep_all=True, device=device) if FastVersion else MTCNN(device=device)
print("MTCNN : {}".format("FastMTCNN" if FastVersion else "MTCNN"))

capture = cv2.VideoCapture(0)

while cv2.waitKey(33) != ord('q'):
    ret, frame = capture.read()
    
    frames = []             #Fast Version
    frames.append(frame)    #Fast Version
    
    faces, _ = mtcnn.detect(frame) if not FastVersion else [mtcnn(frames), 0]
    # print(faces)
    
    try:    
        for face in faces:
            face = np.trunc(face)
            # print("RECT : {}, {}, {}, {}".format(face[0],face[1],face[2],face[3]))
            lu_x, lu_y, rd_x, rd_y  = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            cv2.rectangle(frame, (lu_x, lu_y), (rd_x, rd_y), (255,0,0), 3)
            #cv2.rectangle(frame, (int(face[0]),int(face[1])), (int(face[2]), int(face[3])), (255,0,0), 3)
                
    except Exception as e:
        print("Err : ", e)
    
    cv2.imshow("FACE", frame)

capture.release()
cv2.destroyAllWindows()
