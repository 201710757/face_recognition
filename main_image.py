import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from tqdm import tqdm_notebook as tqdm
import os
import torchvision.transforms as transforms

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(device=device)
model = InceptionResnetV1(pretrained='vggface2').eval()

img = cv2.imread('./test.jpg')
if img is None:
    print("Img Err")
    import sys
    sys.exit()

s_face = 0
faces, _ = mtcnn.detect(img)

model.classify = True
try:    
    for face in faces:
        face = np.trunc(face)
        s_face = img[int(face[1]):int(face[3]), int(face[0]):int(face[2])]
        
        cv2.rectangle(img, (face[0],face[1]), (face[2], face[3]), (255,0,0), 3)
        
        i_c = mtcnn(s_face)
        emb = model(i_c.unsqueeze(0))
        print(emb)
  
except Exception as e:
    print(e)

cv2.imshow("FACE", img)
cv2.imshow("img pos for emb", s_face)
cv2.waitKey()

cv2.destroyAllWindows()
