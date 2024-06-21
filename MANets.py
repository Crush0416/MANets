import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random
import numpy as np
import os


def seeds_init(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  ##  CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    
# ---------------------MyDataset-----------------------
class MyDataset(Dataset):
    def __init__(self, img, label, transform=None):
        super(MyDataset, self).__init__()
        self.img = torch.from_numpy(img).float()
        self.label = torch.from_numpy(label).long()
        self.transform = transform
    
    def __getitem__(self, index):
        img = self.img[index]
        label = self.label[index]
        return img, label
    
    def __len__(self):
        return self.img.shape[0]

def cosine_dist(x_q, x_k):
    
    dots = torch.matmul(x_q, x_v)
    scale = torch.einsum('bhi, bhj -> bhij', (torch.norm(x_q, 2, dim=-1), torch.norm(x_k, 2, dim=-2)))
    dist = dots / scale
    
    return dist

class BackBoneNet(nn.Module):
    """
    ----------feature extraction--------------
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(BackBoneNet, self).__init__()
        self.Layer1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.Layer2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.Layer3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.Layer4 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.Layer5 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.Layer6 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
            # nn.Dropout(0.5)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
               

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channel)
        )       
        
    
    def forward(self, x):
        
        out1 = self.Layer1(x)           #  64*64
        out2 = self.Layer2(out1)        #  32*32
        out3 = self.Layer3(out2)        #  16*16
        out4 = self.Layer4(out3)        #  8*8
        out5 = self.Layer5(out4)        #  4*4
        
        return out3, out4, out5

class MlpBlock(nn.Module):
    # Feed-Forward Block
    def __init__(self, in_dim, mlp_dim, dropout_rate):
        super(MlpBlock, self).__init__()
        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, in_dim)
        self.act = nn.GELU()
        self.LN = nn.LayerNorm(in_dim)
        
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)

        out = self.fc2(out)
        if self.dropout2:
            out = self.dropout2(out)
        
        out = self.LN(out + x)
        return out

class GlobalAveragePooling(nn.Module):
    def __init__(self, in_channel, out_channel, output_size):
        super(GlobalAveragePooling, self).__init__()
        
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)
        self.avg = nn.AdaptiveAvgPool2d(output_size)
    
    def forward(self, x):
        
        out = self.conv(x)
        out = self.avg(out)
        
        return out
        
class LinearGeneral(nn.Module):
    def __init__(self, in_dim, feat_dim):
        super(LinearGeneral, self).__init__()
        self.weight = nn.Parameter(torch.randn(*in_dim, *feat_dim))
        self.bias = nn.Parameter(torch.zeros(*feat_dim))

    def forward(self, x, dims):
        out = torch.tensordot(x, self.weight, dims = dims) + self.bias
        return out                     
        
class SpatialSelfAttention(nn.Module):
    '''
    ----------multi-heads spatial self attention-----------
    '''
    def __init__(self, in_dim, heads, dropout_rate):
        super(SpatialSelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_dim * heads
        self.scale = self.head_dim ** 0.5
        self.LN = nn.LayerNorm(in_dim)

        self.query = nn.Linear(in_dim, in_dim*heads)
        self.key = nn.Linear(in_dim, in_dim*heads)
        self.value = nn.Linear(in_dim, in_dim*heads)
        self.out = nn.Linear(in_dim*heads, in_dim)

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        b, c, _ = x.shape

        #   linear transformation
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        #   reshape for multi-heads attention
        q = q.view(b, c, self.heads, self.head_dim // self.heads)    # [b,c,heads,h*w]
        k = k.view(b, c, self.heads, self.head_dim // self.heads)    # [b,c,heads,h*w]
        v = v.view(b, c, self.heads, self.head_dim // self.heads)    # [b,c,heads,h*w]
        
        #   transpose dimension for matrix multiplication
        q = q.permute(0, 2, 3, 1)    # [b,heads,h*w,c],  head_dim=h*w;
        k = k.permute(0, 2, 3, 1)    # [b,heads,h*w,c]
        v = v.permute(0, 2, 3, 1)    # [b,heads,h*w,c]

        #   scaled dot-product attention 
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale    # [b,heads,head_dim, head_dim]
        attn_weights = F.softmax(scores, dim = -1)
        out = torch.matmul(attn_weights, v)   # [b,heads,h*w,c]
        
        #   concatente and reshape
        out = out.permute(0, 3, 1, 2)    # [b,c,heads,h*w]
        out = out.contiguous().view(b, c, self.head_dim)

        out = self.LN(self.out(out) + x)
        return out, attn_weights        
        
class ChannelSelfAttention(nn.Module):
    '''
          multi-heads channel self attention
    '''
    def __init__(self, in_dim, embedding_dim, heads, dropout_rate):
        super(ChannelSelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = embedding_dim * heads
        self.scale = self.head_dim ** 0.5
        self.LN = nn.LayerNorm(in_dim)

        self.query = nn.Linear(in_dim, embedding_dim*heads)
        self.key = nn.Linear(in_dim, embedding_dim*heads)
        self.value = nn.Linear(in_dim, embedding_dim*heads)
        self.out = nn.Linear(embedding_dim*heads, in_dim)

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        b, c, _ = x.shape

        #   linear transformation
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        #   reshape for multi-heads attention
        q = q.view(b, c, self.heads, self.head_dim // self.heads)
        k = k.view(b, c, self.heads, self.head_dim // self.heads)
        v = v.view(b, c, self.heads, self.head_dim // self.heads)
        
        #   transpose dimension for matrix multiplication
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        #   scaled dot-product attention 
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale    # [b,heads,c,c]
        attn_weights = F.softmax(scores, dim = -1)
        out = torch.matmul(attn_weights, v)
        
        #   concatente and reshape
        out = out.transpose(1, 2).contiguous().view(b, c, self.head_dim)

        out = self.LN(self.out(out) + x)
        return out, attn_weights                                
        
class CrossLevelAttention(nn.Module):
    '''
          multi-heads cross level self attention
    '''
    def __init__(self, in_dim1, in_dim2, in_dim3, qk_embedding, v_embedding, heads, dropout_rate, dist='softmax'):
        super(CrossLevelAttention, self).__init__()
        self.heads = heads
        self.qk_embedding = qk_embedding
        self.v_embedding = v_embedding
        self.scale = (self.qk_embedding * self.heads) ** 0.5
        self.dist = dist
        self.LN = nn.LayerNorm(in_dim3)
        
        self.query = nn.Linear(in_dim1, qk_embedding*heads)
        self.key = nn.Linear(in_dim2, qk_embedding*heads)
        self.value = nn.Linear(in_dim2, v_embedding*heads)
        self.out = nn.Linear(v_embedding*heads, in_dim3)

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
            
    def forward(self, x_q, x_kv, x):
        b, c, _ = x_q.shape
        
        #   linear transformation
        q = self.query(x_q)
        k = self.key(x_kv)
        v = self.value(x_kv)

        #   reshape for multi-heads attention
        q = q.view(b, c, self.heads, self.qk_embedding)   # [b,c,heads,h*w]
        k = k.view(b, c, self.heads, self.qk_embedding)
        v = v.view(b, c, self.heads, self.v_embedding)
        
        #   transpose dimension for matrix multiplication
        q = q.permute(0, 2, 3, 1)     # [b,heads,h*w,c]
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)

        #   attention similarity calculation
        if self.dist == 'consine':
            attn_weights = cosine_dist(q, k.transpose(-2, -1))
        else:     
            scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale    # [b,heads,h*w,h*w]
            attn_weights = F.softmax(scores, dim = -1)
                              
        out = torch.matmul(attn_weights, v)     # [b,heads,h*w,c]
        
        #   concatente and reshape
        out = out.permute(0, 3, 1, 2)    # [b,c,heads,h*w]
        out = out.contiguous().view(b, c, self.heads*self.v_embedding)

        out = self.LN(self.out(out) + x)
        return out, attn_weights
        
        
        
class MANets(nn.Module):
    '''
    ------------the architecture of MANets-----------------
    '''
    def __init__(self, num_classes, heads):
        super(MANets, self).__init__()
        
        self.num_classes = num_classes
        self.heads = heads               
        
        self.backbone = BackBoneNet(in_channel=1, out_channel=64, kernel_size=5, 
                                  stride=1, padding=2)                                
                                  
        self.spatial_attention1 = SpatialSelfAttention(in_dim=16*16, heads=heads, dropout_rate=0.)
        self.spatial_attention2 = SpatialSelfAttention(in_dim=8*8, heads=heads, dropout_rate=0.)
        self.spatial_attention3 = SpatialSelfAttention(in_dim=4*4, heads=heads, dropout_rate=0.)
        
        self.channel_attention1 = ChannelSelfAttention(in_dim=16*16, embedding_dim=16*16, heads=heads, dropout_rate=0.)
        self.channel_attention2 = ChannelSelfAttention(in_dim=8*8, embedding_dim=8*8, heads=heads, dropout_rate=0.)
        self.channel_attention3 = ChannelSelfAttention(in_dim=4*4, embedding_dim=4*4, heads=heads, dropout_rate=0.)
        
        self.cross_attention1 = CrossLevelAttention(in_dim1=16*16, in_dim2=16*16, in_dim3=16*16, qk_embedding=16*16,
                                                    v_embedding=16*16, heads=heads, dropout_rate=0., dist='softmax')
        self.cross_attention2 = CrossLevelAttention(in_dim1=8*8, in_dim2=8*8, in_dim3=8*8, qk_embedding=8*8,
                                                    v_embedding=8*8, heads=heads, dropout_rate=0., dist='softmax')
        self.cross_attention3 = CrossLevelAttention(in_dim1=4*4, in_dim2=4*4, in_dim3=4*4, qk_embedding=4*4,
                                                    v_embedding=4*4, heads=heads, dropout_rate=0., dist='softmax')
        self.cross_attention4 = CrossLevelAttention(in_dim1=16*16, in_dim2=8*8, in_dim3=4*4, qk_embedding=4*4,
                                                    v_embedding=4*4, heads=heads, dropout_rate=0., dist='softmax')
                                                    
        self.mlp1 = MlpBlock(in_dim=16*16, mlp_dim=16*16*2, dropout_rate=0.)
        self.mlp2 = MlpBlock(in_dim=8*8, mlp_dim=8*8*2, dropout_rate=0.)
        self.mlp3 = MlpBlock(in_dim=4*4, mlp_dim=4*4*2, dropout_rate=0.)
        self.mlp4 = MlpBlock(in_dim=4*4, mlp_dim=4*4*2, dropout_rate=0.)

        self.gap = GlobalAveragePooling(in_channel=64, out_channel=128, output_size=(1,1))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=0), ## 1*1
            nn.BatchNorm2d(128),           
            # nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Sequential(
            # nn.Linear(1024, 256),
            # nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        
        #---multi-level feature extraction------
        low_level, middle_level, high_level = self.backbone(x)
        b, c, h, w = high_level.shape

        #---AUX feature stream-----
        # out_aux = self.conv2(high_level)
        
        #---Low Level feature stream-------------       
        out_x = low_level.contiguous().view(b, c, -1)
        out_kv, _ = self.spatial_attention1(out_x)
        out_q, _ = self.channel_attention1(out_x)       
        out_ll, _ = self.cross_attention1(out_q, out_kv, out_x)
        out_ll = self.mlp1(out_ll)
        
        #---Middle Level feature stream-------------
        out_x = middle_level.contiguous().view(b, c, -1)
        out_kv, _ = self.spatial_attention2(out_x)
        out_q, _ = self.channel_attention2(out_x)       
        out_ml, _ = self.cross_attention2(out_q, out_kv, out_x)
        out_ml = self.mlp2(out_ml)
        
        #---High Level feature stream-------------
        out_x = high_level.contiguous().view(b, c, -1)
        out_kv, _ = self.spatial_attention3(out_x)
        out_q, _ = self.channel_attention3(out_x)       
        out_hl, _ = self.cross_attention3(out_q, out_kv, out_x)
        out_hl = self.mlp3(out_hl)               
        
        #---Multi Level feature fusion-------------
        out_q, out_kv, out_x = out_ll, out_ml, out_hl       
        out_fa, _ = self.cross_attention4(out_q, out_kv, out_x)
        out = self.mlp4(out_fa)        
        
        #---conv fusion-----------------
        out = out.contiguous().view(b,c,h,w)
        out = self.conv2(out)
        
        #---Classifier------------------
        out = out.contiguous().view(b, -1)
        out = self.classifier(out)
        
        return out
    
    
    
    
    
    