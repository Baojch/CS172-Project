# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

"""# Import Libraries"""

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from utils.model import select_model
from utils.options import parse_args_function
from utils.dataset import Dataset
import torch.distributed as dist
import torch.multiprocessing as mp
import cv2
import os
from PIL import Image

def visualize_hand_object_single(img_path, joints, boxes, filename=None):
    """
    在图像上绘制手部关节点、手部骨架和物体边界框。
    
    参数:
        img (np): 输入图像路径,
        joints (torch.Tensor): 手部关节点坐标,形状为 [21, 2]。
        boxes (torch.Tensor): 物体边界框坐标,形状为 [8, 2]。
        filename (str, optional): 保存可视化结果的文件名。如果为 None,则不保存。
    """
    # 将 PyTorch 张量转换为 NumPy 数组
    # joints = joints.detach().cpu().numpy()
    # boxes = boxes.detach().cpu().numpy()
    
    #for ground truth
    joints = joints.cpu().numpy()
    boxes = boxes.cpu().numpy()
    
    # 设置手部骨架连接关系
    limbs = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    
    # 设置关节点颜色
    joint_color_code = [[139, 53, 255], [0, 56, 255], [43, 140, 237], [37, 168, 36], [147, 147, 0], [70, 17, 145]]
    
    # 遍历批次
    print(img_path)
    tmp_img = cv2.imread(img_path)
    print(tmp_img.shape)
    img_height, img_width = tmp_img.shape[:2]
    
    # 默认的关节和边界框位置的分辨率
    default_width, default_height = 1920, 1080
    
    # 计算比例
    width_ratio = img_width / default_width
    height_ratio = img_height / default_height
    

    # 对关节点和边界框坐标进行缩放
    joints = joints * [width_ratio, height_ratio]
    boxes = boxes * [width_ratio, height_ratio]
    # 绘制手部关节点
    for joint in joints:
        cv2.circle(tmp_img, tuple(joint.astype(int)), 3, (0, 0, 255), -1)
    
    # 绘制手部骨架
    for limb_num in range(len(limbs)):
        joint1 = joints[limbs[limb_num][0]]
        joint2 = joints[limbs[limb_num][1]]
        color_code_num = limb_num // 4
        limb_color = tuple(joint_color_code[color_code_num])
        cv2.line(tmp_img, tuple(joint1.astype(int)), tuple(joint2.astype(int)), limb_color, 2)
        
    # 绘制物体边界框
    p1, p2, p3, p4, p5, p6, p7, p8 = boxes.astype(int)
    # print(p1, p2, p3, p4, p5, p6, p7, p8)
    
    # 绘制边界框的12条线段
    cv2.line(tmp_img, tuple(p1), tuple(p2), (240, 207, 137), 4)
    cv2.line(tmp_img, tuple(p2), tuple(p4), (240, 207, 137), 4)
    cv2.line(tmp_img, tuple(p3), tuple(p4), (240, 207, 137), 4)
    cv2.line(tmp_img, tuple(p1), tuple(p3), (240, 207, 137), 4)
    
    cv2.line(tmp_img, tuple(p1), tuple(p5), (240, 207, 137), 4)
    cv2.line(tmp_img, tuple(p2), tuple(p6), (240, 207, 137), 4)
    cv2.line(tmp_img, tuple(p3), tuple(p7), (240, 207, 137), 4)
    cv2.line(tmp_img, tuple(p4), tuple(p8), (240, 207, 137), 4)
    
    cv2.line(tmp_img, tuple(p5), tuple(p6), (240, 207, 137), 4)
    cv2.line(tmp_img, tuple(p5), tuple(p7), (240, 207, 137), 4)
    cv2.line(tmp_img, tuple(p7), tuple(p8), (240, 207, 137), 4)
    cv2.line(tmp_img, tuple(p6), tuple(p8), (240, 207, 137), 4)
    
    # 显示或保存可视化结果
    if filename is None:
        cv2.imshow("Hand and Object", tmp_img)
        cv2.waitKey(0)
    else:
        filename_out = filename + "_out" + ".jpg"
        cv2.imwrite(filename_out, tmp_img)
# If your model does not change and your input sizes remain the same - then you may benefit from setting torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    args = parse_args_function()
    device_ids=args.gpu_number
    print("gpus " + ", ".join(map(str, device_ids)))
    print(device_ids)

    """# Load Dataset"""

    root_dir = args.input_file

    #mean = np.array([120.46480086, 107.89070987, 103.00262132])
    #std = np.array([5.9113948 , 5.22646725, 5.47829601])

    """# Model"""

    use_cuda = False
    if args.gpu:
        use_cuda = True

    model = select_model(args.model_def)

    if use_cuda and torch.cuda.is_available():
        print('Using GPU')
        # 模型加载到设备0
        model = model.cuda(device_ids[0])
        # 指定要用到的设备
        model = nn.DataParallel(model, device_ids=device_ids)
        

    """# Load Snapshot"""

    if args.pretrained_model != '':
        model.load_state_dict(torch.load(args.pretrained_model))
        losses = np.load(args.pretrained_model[:-4] + '-losses.npy').tolist()
        start = len(losses)
    elif args.resume != '':
        print("resume from:"+args.resume)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint)
        losses = np.load(args.resume[:-4] + '-losses.npy').tolist()
        start = len(losses)
    else:
        losses = []
        start = 0

    """# Optimizer"""

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_step_gamma)
    scheduler.last_epoch = start
    lambda_1 = 0.01
    lambda_2 = 1

    
    """# Test"""

    if args.test:
        print('Begin testing imgs...')
        # 定义transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        image_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff')):
                    image_paths.append(os.path.join(root, file))
        print(image_paths)
        
        # 调整所有图片大小并保存
        target_size = (1920, 1080)  # (width, height)

        for image_path in image_paths:
            with Image.open(image_path) as image:
                # 调整图片大小
                resized_image = image.resize(target_size, Image.LANCZOS)
                # 保存图片到原路径
                resized_image.save(image_path)
        
        # 读取并转换所有图片
        images = []
        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')
            image = transform(image)
            images.append(image)

        # 转换成[batch, 3, 224, 224]的张量
        batch_tensor = torch.stack(images)
        print(batch_tensor.shape)
        outputs2d_init, outputs2d, outputs3d = model(batch_tensor)
        for i in range(len(image_paths)):
            file_name = "./output/"+os.path.basename(image_paths[i]).replace(".jpg", "").replace(".jpeg", "")
            print(file_name)
            visualize_hand_object_single(image_paths[i], outputs2d[i, :21, :].detach(), outputs2d[i, 21:, :].detach(), file_name)
    