
""" 
       testing on Multi-Level Attention Networks---MANets
"""

# import
from MANets import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import scipy.io

#---------------------__Main__-----------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print ('GPU is true')
    print('cuda version: {}'.format(torch.version.cuda))
else:
    print('CPU is true')
    
#  hyperparameter setting 
PATH = './models/EOC-7/100%/fullmodel_12Ep_0.9792Acc.pth'      ## model path   
batch_size = 32

#----------------DataLoader-----------------
train_dataset = scipy.io.loadmat('../../Datasets/MANets-Datasets/EOC/train/data_train_128_100%.mat')
test_dataset  = scipy.io.loadmat('../../Datasets/MANets-Datasets/EOC/test/data_test_128_dv_vv.mat')

traindata = train_dataset['data_am']
trainlabel = train_dataset['label'].squeeze()    ##  label必须是一维向量

testdata = test_dataset['data_am']
testlabel = test_dataset['label'].squeeze()

train_dataset = MyDataset(img=traindata, label=trainlabel, transform=transforms.ToTensor())
test_dataset  = MyDataset(img=testdata, label=testlabel, transform=transforms.ToTensor())
print('train data size: {}'.format(train_dataset.img.shape[0]))
print('test data size: {}'.format(test_dataset.img.shape[0]))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#---------------------model loading------------------
model = torch.load(PATH)
criterion = nn.CrossEntropyLoss()
#   计算模型参数
param_num = sum([param.nelement() for param in model.parameters()])
print('The number of model parameters: {:.4f} M'.format(param_num/1e6))

#-------------------------testing-----------------------
print('--------------test starting...-----------------')

model.eval()
with torch.no_grad():
    labels = []
    labels_pre = []

    total_loss = 0
    total = 0
    correct = 0
    for batch_idx, (image, label) in enumerate(train_loader):
        
        images   = image.to(device)   
        label   = label.to(device)
                
        output = model(images)
        loss = criterion(output, label)          
        
        total_loss += loss.item()                               
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        # labels.append(label)
        # labels_pre.append(predicted)
        labels.extend(label.data.cpu().numpy())      # data form GPU to CPU
        labels_pre.extend(predicted.data.cpu().numpy())

    print('---------------------------training-----------------------------')    
    print('correct number : {}, train data number : {}, Accuracy : {:.4f}, train loss: {:.6f}'.format(correct, total, 100 * correct / total, total_loss))   
    matrix = confusion_matrix(labels, labels_pre)
    print('############ confusion matrix ########### \n', matrix)
    
#----------------Testing----------------
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    temp_loss = 0
    labels = []
    labels_pre = []
    for image, label in test_loader:
        image   = image.to(device)		
        label   = label.to(device)

        output = model(image)
        loss = criterion(output, label)
        temp_loss += loss.item()

        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        # labels.append(label)
        # labels_pre.append(predicted)
        labels.extend(label.data.cpu().numpy())      # data form GPU to CPU
        labels_pre.extend(predicted.data.cpu().numpy())
    print('---------------------------testing---------------------------')  
    print('correct number : {}, test data number : {}, Accuracy : {:.4f}, test loss: {:.6f}'.format(correct, total, 100 * correct / total, temp_loss))

    matrix = confusion_matrix(labels, labels_pre)
    print('############ confusion matrix ########### \n', matrix)
    # scipy.io.savemat('./results/confusion_matrix_EOC7.mat',{'confusion_matrix':matrix,'label':labels, 'label_predict':labels_pre})
    print('----------------------------------------------------------------\n')
    print('****************************************************************')

          

    


















 