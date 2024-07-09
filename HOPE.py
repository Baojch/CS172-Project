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

def visualize_hand_object(img_path, joints, boxes, filename=None):
    """
    在图像上绘制手部关节点、手部骨架和物体边界框。
    
    参数:
        img (np): 输入图像路径,形状为 [batch_size, _]。
        joints (torch.Tensor): 手部关节点坐标,形状为 [batch_size*gpu_num, 21, 2]。
        boxes (torch.Tensor): 物体边界框坐标,形状为 [batch_size*gpu_num, 8, 2]。
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
    for i in range(joints.shape[0]):
        tmp_img = cv2.imread(img_path[i])
        # 绘制手部关节点
        for joint in joints[i]:
            cv2.circle(tmp_img, tuple(joint.astype(int)), 3, (0, 0, 255), -1)
        
        # 绘制手部骨架
        for limb_num in range(len(limbs)):
            joint1 = joints[i][limbs[limb_num][0]]
            joint2 = joints[i][limbs[limb_num][1]]
            color_code_num = limb_num // 4
            limb_color = tuple(joint_color_code[color_code_num])
            cv2.line(tmp_img, tuple(joint1.astype(int)), tuple(joint2.astype(int)), limb_color, 2)
        
        # 绘制物体边界框
        p1, p2, p3, p4, p5, p6, p7, p8 = boxes[i].astype(int)
        # print(p1, p2, p3, p4, p5, p6, p7, p8)
        
        # 绘制边界框的12条线段
        cv2.line(tmp_img, tuple(p1), tuple(p2), (0, 255, 0), 2)
        cv2.line(tmp_img, tuple(p2), tuple(p4), (0, 255, 0), 2)
        cv2.line(tmp_img, tuple(p3), tuple(p4), (0, 255, 0), 2)
        cv2.line(tmp_img, tuple(p1), tuple(p3), (0, 255, 0), 2)
        
        cv2.line(tmp_img, tuple(p1), tuple(p5), (0, 255, 0), 2)
        cv2.line(tmp_img, tuple(p2), tuple(p6), (0, 255, 0), 2)
        cv2.line(tmp_img, tuple(p3), tuple(p7), (0, 255, 0), 2)
        cv2.line(tmp_img, tuple(p4), tuple(p8), (0, 255, 0), 2)
        
        cv2.line(tmp_img, tuple(p5), tuple(p6), (0, 255, 0), 2)
        cv2.line(tmp_img, tuple(p5), tuple(p7), (0, 255, 0), 2)
        cv2.line(tmp_img, tuple(p7), tuple(p8), (0, 255, 0), 2)
        cv2.line(tmp_img, tuple(p6), tuple(p8), (0, 255, 0), 2)
        
        # 显示或保存可视化结果
        if filename is None:
            cv2.imshow("Hand and Object", tmp_img)
            cv2.waitKey(0)
        else:
            filename_out = filename + "_" +str(i) + ".jpg"
            cv2.imwrite(filename_out, tmp_img)
# If your model does not change and your input sizes remain the same - then you may benefit from setting torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    args = parse_args_function()
    device_ids=args.gpu_number
    print("gpus " + ", ".join(map(str, device_ids)))
    print(device_ids)

    """# Load Dataset"""

    root = args.input_file

    #mean = np.array([120.46480086, 107.89070987, 103.00262132])
    #std = np.array([5.9113948 , 5.22646725, 5.47829601])

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])

    if args.train:
        trainset = Dataset(root=root, load_set='train', transform=transform, include_path=False)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size * len(device_ids), shuffle=True, num_workers=16, pin_memory=True)
        
        print('Train files loaded')

    if args.val:
        valset = Dataset(root=root, load_set='val', transform=transform, include_path=False)
        valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size * len(device_ids), shuffle=False, num_workers=8, pin_memory=True)
        
        print('Validation files loaded')

    if args.test:
        testset = Dataset(root=root, load_set='test', transform=transform, include_path=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size * len(device_ids), shuffle=False, num_workers=8, pin_memory=True)
        
        print('Test files loaded')

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

    """# Train"""

    if args.train:
        print('Begin training the network...')
        
        for epoch in range(start, args.num_iterations):  # loop over the dataset multiple times
        
            running_loss = 0.0
            train_loss = 0.0
            loss_2d = 0.0
            loss_3d = 0.0
            for i, tr_data in enumerate(trainloader):
                # get the inputs
                inputs, labels2d, labels3d = tr_data
        
                # wrap them in Variable
                inputs = Variable(inputs)
                labels2d = Variable(labels2d)
                labels3d = Variable(labels3d)
                
                if use_cuda and torch.cuda.is_available():
                    inputs = inputs.float().cuda(device=args.gpu_number[0])
                    labels2d = labels2d.float().cuda(device=args.gpu_number[0])
                    labels3d = labels3d.float().cuda(device=args.gpu_number[0])
        
                # zero the parameter gradients
                optimizer.zero_grad()
        
                # forward + backward + optimize
                outputs2d_init, outputs2d, outputs3d = model(inputs)
                loss2d_init = criterion(outputs2d_init, labels2d)
                loss2d = criterion(outputs2d, labels2d)
                loss3d = criterion(outputs3d, labels3d)
                loss = (lambda_1)*loss2d_init + (lambda_1)*loss2d + (lambda_2)*loss3d
                # some finetuning for the root joint
                # root_loss = criterion(outputs2d[:, 0, :], labels2d[:, 0, :])
                # loss = (lambda_1)*loss2d_init + (lambda_1)*loss2d + (lambda_2)*loss3d + root_loss
                loss.backward()
                optimizer.step()
                
                # print statistics
                loss_2d += loss2d.data
                loss_3d += loss3d.data
                running_loss += loss.data
                train_loss += loss.data
                if (i+1) % args.log_batch == 0:    # print every log_iter mini-batches
                    print('[%d, %5d] loss: %.5f, loss2d: %.3f,loss3d: %.3f' % (epoch + 1, i + 1, running_loss / args.log_batch, loss_2d / args.log_batch, loss_3d / args.log_batch))
                    running_loss = 0.0
                    loss_2d = 0.0
                    loss_3d = 0.0
                    
            if args.val and (epoch+1) % args.val_epoch == 0:
                val_loss = 0.0
                for v, val_data in enumerate(valloader):
                    # get the inputs
                    inputs, labels2d, labels3d = val_data
                    
                    # wrap them in Variable
                    inputs = Variable(inputs)
                    labels2d = Variable(labels2d)
                    labels3d = Variable(labels3d)
            
                    if use_cuda and torch.cuda.is_available():
                        inputs = inputs.float().cuda(device=args.gpu_number[0])
                        labels2d = labels2d.float().cuda(device=args.gpu_number[0])
                        labels3d = labels3d.float().cuda(device=args.gpu_number[0])
            
                    outputs2d_init, outputs2d, outputs3d = model(inputs)
                    
                    loss2d_init = criterion(outputs2d_init, labels2d)
                    loss2d = criterion(outputs2d, labels2d)
                    loss3d = criterion(outputs3d, labels3d)
                    loss = (lambda_1)*loss2d_init + (lambda_1)*loss2d + (lambda_2)*loss3d
                    val_loss += loss.data
                print('val error: %.5f' % (val_loss / (v+1)))
            losses.append((train_loss / (i+1)).cpu().numpy())
            
            if (epoch+1) % args.snapshot_epoch == 0:
                torch.save(model.state_dict(), args.output_file+str(epoch+1)+'.pkl')
                np.save(args.output_file+str(epoch+1)+'-losses.npy', np.array(losses))

            # Decay Learning Rate
            scheduler.step()
        
        print('Finished Training')

    """# Test"""

    if args.test:
        print('Begin testing the network...')
        
        running_loss = 0.0
        for i, ts_data in enumerate(testloader):
            # get the inputs
            inputs, labels2d, labels3d, image_path = ts_data
            print(inputs.shape)
            # print(np.array(image_path)[0]) #图片路径           
            # wrap them in Variable
            inputs = Variable(inputs)
            labels2d = Variable(labels2d)
            labels3d = Variable(labels3d)

            if use_cuda and torch.cuda.is_available():
                inputs = inputs.float().cuda(device=args.gpu_number[0])
                labels2d = labels2d.float().cuda(device=args.gpu_number[0])
                labels3d = labels3d.float().cuda(device=args.gpu_number[0])
            # visualize_hand_object(np.array(image_path), labels2d[:, :21, :], labels2d[:, 21:, :], filename="./output/visualization_gt")
            outputs2d_init, outputs2d, outputs3d = model(inputs)
            visualize_hand_object(np.array(image_path), outputs2d[:, :21, :].detach(), outputs2d[:, 21:, :].detach(), filename="./output/visualization")
            loss = criterion(outputs2d, labels2d)
            # loss = criterion(outputs3d, labels3d)
            running_loss += loss.data
        print('test error: %.5f' % (running_loss / (i+1)))