import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from tqdm import tqdm_notebook as tqdm
import os

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(device=device)
model = InceptionResnetV1(pretrained='vggface2').eval()
#capture = cv2.VideoCapture(0)

#while cv2.waitKey(33) != ord('q'):
#    ret, frame = capture.read()
img = cv2.imread('./test.jpg')
if img is None:
    print("Img Err")
    import sys
    sys.exit()

s_face = 0
faces, _ = mtcnn.detect(img)

try:    
    for face in faces:
        face = np.trunc(face)
        s_face = img[int(face[1]):int(face[3]), int(face[0]):int(face[2])]
        cv2.rectangle(img, (face[0],face[1]), (face[2], face[3]), (255,0,0), 3)
        
        y = model.predict(s_face)
        embedding = y[0]
        print("Embedding : ", embedding)
except:
    pass

cv2.imshow("FACE", img)
cv2.imshow("em", s_face)
cv2.waitKey()

cv2.destroyAllWindows()
