from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(device=device)
model = InceptionResnetV1(pretrained='vggface2').eval()

img = Image.open('./test.jpg')

img_cropped = mtcnn(img)
model.classify = True
img_embedding = model(img_cropped.unsqueeze(0))

print(img_embedding)




