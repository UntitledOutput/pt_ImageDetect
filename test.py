import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import train
import time

#import intel_npu_acceleration_library as intel_npu

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img.cpu()
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('output.png')  # Save the image to a file
    print("\tLook at the picture...")
    time.sleep(10)


def color_percentage(value):

    if value <= 20:
        color = "\033[91m"
    elif value <= 40:
        color = "\033[33m" 
    elif value <= 60:
        color = "\033[93m"  
    elif value <= 80:
        color = "\033[92m"  
    else:
        color = "\033[96m" 

    reset = "\033[0m" 
    return f"{color}{value}%{reset}"

def separate():
    print("------------------------------------")

def test():

    model = train.NeuralNetwork()
    model.load_state_dict(torch.load('cifar_net.pth',weights_only=True))
    model.eval()


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("Using device: ", torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU")
    #print("Using device: ", intel_npu.get_device_name(device) if intel_npu.is_available() else "CPU")

    separate()

    model.to(device)

    testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=train.transform)
    testloader = torch.utils.data.DataLoader(testset,batch_size=train.batch_size,shuffle=True,num_workers=2)

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    images = images.to(device)

    print("Doing first predictions")

    #separate()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('\tChosen classes:\t', ' '.join(
        f'{train.classes[labels[j]]}' 
        for j in range(train.batch_size)
        ))

    #separate()

    outputs = model(images)

    _, predicted = torch.max(outputs, 1)

    reset = "\033[0m" 
    print('\tPredicted:\t', ' '.join(
        f'\033[92m{train.classes[predicted[j]]}{reset}' if train.classes[predicted[j]] == train.classes[labels[j]] 
        else f'\033[91m{train.classes[predicted[j]]}{reset}'
        for j in range(train.batch_size)
    ))
    
    separate()

    print("Testing the model on 10000 test images")

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'\tAccuracy of the network on the 10000 test images: {color_percentage(100 * correct // total)} %')

    separate()

    print("Testing the model on each class")

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in train.classes}
    total_pred = {classname: 0 for classname in train.classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[train.classes[label]] += 1
                total_pred[train.classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'\tAccuracy for class: {classname:5s} is {color_percentage( accuracy )} %')

    separate()

    del dataiter, images, labels, outputs, predicted, correct, total, correct_pred, total_pred
    del model, testset, testloader, device, accuracy, classname, correct_count

    val = input("Do you want to retrain the model? (y/n)")
    if val == 'y':
        separate()
        train.train()
        test()
    else:
        pass
if __name__ == '__main__':
    test()