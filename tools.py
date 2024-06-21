import torch
import torch.nn as nn

class CrossEntorpyLoss2d(nn.Module):
    def __init__(self):
        super(Class_AzimuthLoss, self).__init__()
        self.alpha = torch.nn.Parameter(torch.FloatTensor(1), require_grad=True)
        self.alpha.data.fill_(1)     #  初始化为1 
        self.FAFLoss = nn.CrossEntropyLoss()
        self.GVFLoss = nn.CrossEntropyLoss()
        
    def forward(self, output_fa, output_aux, label):
        
        Loss_fa = self.FAFLoss(output_fa, label)
        Loss_aux = self.alpha*self.GVFLoss(output_aux, azimuth)
        Loss = Loss_fa + self.alpha * Loss_aux
        
        return Loss