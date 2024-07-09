# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from models.graphunet import GraphUNet
from models.resnet import resnet50, resnet10
from torch_geometric.nn import GATConv
from transformers import ViTModel


class HopeNetModify(nn.Module):

    def __init__(self):
        super(HopeNetModify, self).__init__()
        self.vit = ViTModel.from_pretrained('./models/vit-base-patch16-224')
        self.fc = nn.Linear(self.vit.config.hidden_size, 29*2)
        self.dropout = nn.Dropout(p=0.3)
        self.graphnet1 = GATConv(770, 256, heads=4, concat=True)
        self.graphnet2 = GATConv(256 * 4, 16, heads=4, concat=True)
        self.graphnet3 = GATConv(16 * 4, 2)
        self.graphunet = GraphUNet(in_features=2, out_features=3)
        
        # joints for hand: 0-20, bounding box 21-28
        limbs = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], 
                 [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], 
                 [15, 16], [0, 17], [17, 18], [18, 19], [19, 20],
                 [21, 22], [22, 24], [23, 24], [21, 23], [21, 25], [22, 26], 
                 [23, 27], [24, 28], [25, 26], [25, 27], [26, 28], [27, 28]]

        edges = []
        for limb in limbs:
            edges.append([limb[0], limb[1]])
            edges.append([limb[1], limb[0]])

        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    def forward(self, x):
        device = x.device
        # print(device)
        self.edge_index = self.edge_index.to(device)
        batch_size = x.size(0)
        features = self.vit(x).last_hidden_state
        features = features.mean(dim=1)  # Pooling to get a single vector representation
        features = self.dropout(features)
        points2D_init = self.fc(features)
        points2D_init = points2D_init.view(-1, 29, 2)
        features = features.unsqueeze(1).repeat(1, 29, 1)
        in_features = torch.cat([points2D_init, features], dim=2)
        points2D_list = []
        for i in range(batch_size):
            out = self.graphnet1(in_features[i], self.edge_index)
            out = self.graphnet2(out, self.edge_index)
            out = self.graphnet3(out, self.edge_index)
            points2D_list.append(out)
        points2D = torch.stack(points2D_list)
        
        points3D = self.graphunet(points2D)
        return points2D_init, points2D, points3D
