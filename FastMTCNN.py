from facenet_pytorch import MTCNN
import cv2


class FastMTCNN(object):
    def __init__(self, stride=4, resize=1, *args, **kwargs):
        self.stride = stride
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)
        
    def __call__(self, frames):
        if self.resize != 1:
            frames = [
                cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))
                    for f in frames
            ]
                      
        boxes, probs = self.mtcnn.detect(frames[::self.stride])

        faces = []
        for i, frame in enumerate(frames):
            box_ind = int(i / self.stride)
            try:
                if boxes[box_ind] is None:
                    continue
            
                for box in boxes[box_ind]:
                    box = [int(b) for b in box]
                    faces.append([box[0], box[1], box[2], box[3]])
                    #print("BOX : {}, {}, {} ,{}".format(box[0], box[1], box[2], box[3]))
            except:
                pass
        return faces
