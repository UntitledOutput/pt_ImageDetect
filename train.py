

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import download_data


datapath = "./data/ceos"

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
batch_size = 12

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        width = 24

        self.conv1 = nn.Conv2d(3,width,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(width,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
def save_checkpoint(model, optimizer, epoch, loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, 'cifar_net_train.pth')
    print("Checkpoint saved successfully. (Epoch {0})".format(epoch+1))

def load_checkpoint(filepath, model, optimizer):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded successfully from {filepath}. Resuming from epoch {epoch}.")
        return epoch, loss
    else:
        print(f"No checkpoint found at {filepath}. Starting from scratch.")
        return 0, None

def train():

    download_data.download_data(datapath,classes)
    exit(1)

    if (not os.path.exists(datapath)):
        os.makedirs(datapath)
    if (not os.listdir(datapath)):
        download_data.download_data(datapath,classes)
        exit(1)


    print("Training with classes: ", classes)

    trainset = torchvision.datasets.ImageFolder(root=datapath, train=True,download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = NeuralNetwork()
    
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

    start_epoch, _ = load_checkpoint('cifar_net_train.pth', net, optimizer)

    print("Starting training...")

    for epoch in range(start_epoch,start_epoch+16): 

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


            running_loss += loss.item()
            if i % 2000 == 1999: 
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        
        save_checkpoint(net, optimizer, epoch, running_loss)

    print('Finished Training')


    torch.save(net.state_dict(), './cifar_net.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, './cifar_net_train.pth')


    pass

if __name__ == '__main__':
    train()  # Call the train function to start training the model.
    pass
