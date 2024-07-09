from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

"""# Import Libraries"""

import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import cv2
from PIL import Image
from HOPE import visualize_hand_object

def demo(model, use_cuda, device, \
          input_path = './input/input.mp4', temp_path = './output_temp/', output_path = './output/output.mp4',\
          width = 1920, height = 1080, fps = 30, fourcc = cv2.VideoWriter_fourcc(*'mp4v')):
    cap = cv2.VideoCapture(input_path)
    res, frame = cap.read()
    frames = []
    frame_cnt = 0
    frames_path = []
    while(res):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.array(Image.fromarray(frame).resize((width, height), Image.LANCZOS))
        path = temp_path + 'temp_img_' + str(frame_cnt) + '.jpg'
        frames_path.append(path)
        cv2.imwrite(path, frame)
        frames.append(frame)
        res, frame = cap.read()
        frame_cnt += 1
        
    if(frame_cnt == 0):
        raise RuntimeError("No frame in the video!")

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])
    
    dataset = []
    for f in frames:
        f = Image.fromarray(f)
        dataset.append(transform(f))

    videowriter = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    img_id = 0
    for img in dataset:
        with torch.no_grad():
            img = img.unsqueeze(0)
            inputs = Variable(img)

            if use_cuda and torch.cuda.is_available():
                inputs = inputs.float().cuda(device = device)
            outputs2d_init, outputs2d, outputs3d = model(inputs)
            visualize_hand_object(np.array([frames_path[img_id]]),\
                                   outputs2d[:, :21, :].detach(), outputs2d[:, 21:, :].detach(),\
                                   filename = temp_path + "demo_visualization")
            vis_img_path = temp_path + "demo_visualization_0.jpg"
            img = cv2.imread(vis_img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #print(np.asarray(img).shape)
            videowriter.write(img)
        img_id += 1
    #print(img_id)

    videowriter.release()

    print('done')

