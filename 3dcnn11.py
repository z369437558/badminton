import torch
from torchvision import transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
 
# prepare dataset
 
batch_size = 50
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
input_data=torch.randn(50,3,10,256,256) #batch size*chanel*frames*w*h
# design model using class
# input_label=torch.rand(50)*100
input_label=torch.randint(3, (50,))
train_ids = TensorDataset(input_data, input_label) 
train_loader = DataLoader(dataset=train_ids, batch_size=batch_size, shuffle=True)
# test=torch.randn(20,3,10,256,256) 
test = input_data
# test_label=torch.rand(50)*100
test_label = input_label
test_ids = TensorDataset(test, test_label) 
test_loader = DataLoader(dataset=test_ids, batch_size=batch_size, shuffle=True)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv3d(3,3,(3,7,7), stride=1, padding=0)#输入通道数，输出通道数，卷积核
        # self.conv2 = torch.nn.Conv3d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool3d(2)
        self.fc1 = torch.nn.Linear(187500, 500) #输入维度，输出维度
        self.fc2 = torch.nn.Linear(500, 3)

    def forward(self, x):
        # flatten data from (n,1,28,28) to (n, 784)
        
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        # x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1) # -1 此处自动算出的是320
        print("x.shape", x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
 
        return x

model = Net()
device = "cpu"
model.to(device)
 
# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
 
# training cycle forward, backward, update
 
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device,dtype=torch.int64)
        optimizer.zero_grad()
        #print(inputs.shape)
        outputs = model(inputs)
        print(target)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
 
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch+1, batch_idx+1, running_loss/300))
            running_loss = 0.0
 
 
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device,dtype=torch.int64)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %% ' % (100*correct/total))
    return correct/total


if __name__ == '__main__':
    epoch_list = []
    acc_list = []
    
    for epoch in range(20):
        train(epoch)
        acc = test()
        epoch_list.append(epoch)
        acc_list.append(acc)
    
    plt.plot(epoch_list, acc_list)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
 
    