import torch
import torch.nn as nn
import torch.nn.functional as F

class Graph:
    """MediaPipe 33-keypoint skeleton adjacency matrix"""
    def __init__(self):
        self.num_node = 33
        # Define edges (pairs of connected joints)
        # Using simplified BlazePose connections
        self.edges = [
            (0,1),(1,2),(2,3),(3,7),  # right arm
            (0,4),(4,5),(5,6),(6,8),  # left arm
            (9,10),(11,12),
            (11,13),(13,15),(15,17),(15,19),(15,21),
            (12,14),(14,16),(16,18),(16,20),(16,22),
            (23,24),(11,23),(12,24),
            (23,25),(24,26),(25,27),(26,28),(27,29),(28,30),(29,31),(30,32)
        ]
        self.A = self.get_adjacency()

    def get_adjacency(self):
        V = self.num_node
        A = torch.eye(V, dtype=torch.float32)
        for i,j in self.edges:
            A[i,j] = 1.0
            A[j,i] = 1.0
        # normalize adjacency (symmetric normalization)
        D = A.sum(1)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
        D_mat = torch.diag(D_inv_sqrt)
        A_norm = D_mat @ A @ D_mat
        return A_norm, self.edges

    def get_adjacency(self):
        A = torch.eye(self.num_node)
        for i,j in self.edges:
            A[i,j] = 1
            A[j,i] = 1
        # Normalize adjacency
        D = A.sum(1)
        D = torch.diag(torch.pow(D, -0.5))
        A_norm = D @ A @ D
        return A_norm

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super().__init__()
        self.A = A
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=(1, A.size(0)))
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9,1), padding=(4,0), stride=(stride,1)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.3)
        )
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride,1)),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (N, C, T, V)
        res = self.residual(x)
        # Graph convolution
        x = torch.einsum('nctv,vw->nctw', x, self.A)
        x = self.gcn(x)
        x = self.tcn(x)
        x = x + res
        return self.relu(x)

class STGCN(nn.Module):
    def __init__(self, num_class=4, num_point=33, num_person=1, in_channels=3):
        super().__init__()
        self.graph = Graph()
        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.st_gcn_networks = nn.ModuleList((
            STGCNBlock(in_channels, 64, A, residual=False),
            STGCNBlock(64, 64, A),
            STGCNBlock(64, 128, A, stride=2),
            STGCNBlock(128, 256, A, stride=2),
        ))
        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, 0.01)

    def forward(self, x):
        # x: (N, C, T, V, M)  [Batch, Channels, Frames, Joints, Persons]
        N, C, T, V, M = x.size()
        x = x.permute(0,4,1,2,3).contiguous().view(N, M * C * V, T)
        x = self.data_bn(x)
        x = x.view(N, M * C, T, V)
        for gcn in self.st_gcn_networks:
            x = gcn(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1)
        return self.fc(x)
