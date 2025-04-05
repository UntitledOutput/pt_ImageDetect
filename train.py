import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet18 
from PIL import Image

import os
import download_data
import warnings
warnings.filterwarnings('ignore')


datapath = "data/ceos/"

classes = tuple([name for name in os.listdir(datapath+"/train/") if os.path.isdir(os.path.join(datapath+"/train/", name))]) if os.path.exists(datapath+"/train/") else ()

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
batch_size = 128
workers = 12  # Number of worker threads for DataLoader

def collate_fn(batch):
  return {
      'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
      'labels': torch.tensor([x['labels'] for x in batch])
}

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        width = 24

        self.conv1 = nn.Conv2d(3,width,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(width,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,len(classes))

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

    if (not os.path.exists(datapath)):
        os.makedirs(datapath)
    if (not os.listdir(datapath)):

        with open('celebrities.txt', 'r') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]


        download_data.download_data(datapath,classes)
        #exit(1)

    classes = tuple([name for name in os.listdir(datapath+"/train/") if os.path.isdir(os.path.join(datapath+"/train/", name))])
    download_data.check_data_images(datapath)  # Check and clean images in train/test folders


    print("Training with classes: ", classes)

    #trainset = trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True,transform=transform)
    trainset = torchvision.datasets.ImageFolder(root=datapath+"/train/", transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=workers,pin_memory=True,persistent_workers=True)



    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = resnet18(num_classes=len(classes))  # Load ResNet18 model without pretrained weights

    if torch.cuda.is_available():
        # Use CUDA-specific optimizations
        net.to(device, memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True  # Enable benchmark mode for faster training on GPUs
        torch.backends.cuda.matmul.allow_tf32 = True  # Allows TensorFloat-32 (faster matmul)
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # Allows FP16 optimizations
    else:
        # Use CPU
        net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.001)

    start_epoch, _ = load_checkpoint('cifar_net_train.pth', net, optimizer)

    print("Using device: ", torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU")
    print("Starting training...")

    # AMP (Automatic Mixed Precision) scaler
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    epochCount = 8

    for epoch in range(start_epoch, start_epoch + epochCount): 
        running_loss = 0.0
        total_batches = len(trainloader)
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            # Enable AMP only if CUDA is available
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                # Forward pass
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = running_loss / total_batches
        print(f'Epoch [{epoch + 1}] Average Loss: {avg_loss:.15f} ( {start_epoch + (epoch - start_epoch) + 1}/{start_epoch + epochCount} )')

        save_checkpoint(net, optimizer, epoch, avg_loss)

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
