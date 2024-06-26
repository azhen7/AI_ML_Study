import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import resnet18
import torch.optim as optim

MODEL_WEIGHTS_PATH = "model_weights.pth"

# load CIFAR-10
transform_train = transforms.Compose(
    [transforms.Resize((32,32)),  #resises the image so it can be perfect for our model.
     transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
     transforms.RandomRotation(10),     #Rotates the image to a specified angel
     transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform = transforms.Compose([transforms.Resize((32,32)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])
batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = [
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def train(model, trainloader, trainset, optimizer, criterion):
    model.train()
    correct = 0
    for _, data in enumerate(trainloader, 0):
         # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs = inputs.to(torch.device('cuda'))
        labels = labels.to(torch.device('cuda'))

        # zero the parameter gradients
        optimizer.zero_grad()

        # get outputs
        outputs = model(inputs)

        # compute loss, go backwards
        loss = criterion(outputs, labels)

        # get predictions
        predictions = torch.argmax(outputs, 1)

        # get number of correct predictions
        correct += (predictions == labels).sum().item()

        # backpropagation
        loss.backward()

        # update weights
        optimizer.step()
    
    accuracy = 100.0 * correct / len(trainset)
    return accuracy

def validate(model, testloader, testset):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(torch.device('cuda'))
            labels = labels.to(torch.device('cuda'))
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy


def model():
    # model
    net = resnet18(num_classes=10).cuda()

    # optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # epochs
    EPOCHS = 100

    # train
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        if epoch == 20:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001
        # correct = 0
        # for _, data in enumerate(trainloader, 0):
        #     # get the inputs; data is a list of [inputs, labels]
        #     inputs, labels = data

        #     inputs = inputs.to(torch.device('cuda'))
        #     labels = labels.to(torch.device('cuda'))

        #     # zero the parameter gradients
        #     optimizer.zero_grad()

        #     # get outputs
        #     outputs = net(inputs)

        #     # compute loss, go backwards
        #     loss = criterion(outputs, labels)

        #     # get predictions
        #     predictions = torch.argmax(outputs, 1)

        #     # get number of correct predictions
        #     correct += (predictions == labels).float().sum()

        #     # backpropagation
        #     loss.backward()

        #     # update weights
        #     optimizer.step()
        
        # # calculate accuracy
        # accuracy = 100 * correct / len(trainset)
        # print(f"Epoch {epoch} done with accuracy {accuracy}%")
        print(f"--------------- Epoch {epoch + 1} ---------------")
        train_epoch_acc = train(
            net,
            trainloader,
            trainset,
            optimizer,
            criterion
        )
        print(f"{epoch + 1}: Training accuracy: {train_epoch_acc:.3f}%")
        valid_epoch_acc = validate(
            net,
            testloader,
            testset
        )
        print(f"{epoch + 1}: Validation accuracy: {valid_epoch_acc:.3f}%")
    
    torch.save(net.state_dict(), MODEL_WEIGHTS_PATH) # save model weights in a file
    print('Finished Training')

    # testing
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(torch.device('cuda'))
            labels = labels.to(torch.device('cuda'))
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(torch.device('cuda'))
            labels = labels.to(torch.device('cuda'))
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

def load():
    model = resnet18(num_classes=10).cuda()
    model

if __name__ == '__main__':
    model()