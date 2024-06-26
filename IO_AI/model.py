import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import resnet18
import torch.optim as optim

MODEL_WEIGHTS_PATH = "model_weights.pth"

# load CIFAR-10
transform_train = transforms.Compose([
    transforms.Resize((32, 32)),  # resizes the image so it can be perfect for our model.
    transforms.RandomHorizontalFlip(), # flips the image w.r.t horizontal axis
    transforms.RandomRotation(10), # rotates the image to a random angle
    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), # performs actions like zooms, change shear angles, etc.
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize image between -1, 1
])

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize image between -1, 1
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

# trains the model for 1 epoch
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

# validates the model on the test data
def validate(model, testloader):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad(): # torch.no_grad() turns off gradients
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
        # update learning rate to 10x reduction from initial
        if epoch == 20:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001
        
        print(f"--------------- Epoch {epoch + 1} ---------------")

        # training accuracy
        train_epoch_acc = train(
            net,
            trainloader,
            trainset,
            optimizer,
            criterion
        )
        print(f"{epoch + 1}: Training accuracy: {train_epoch_acc:.3f}%")

        # validation accuracy
        valid_epoch_acc = validate(
            net,
            testloader
        )
        print(f"{epoch + 1}: Validation accuracy: {valid_epoch_acc:.3f}%")
    
    torch.save(net.state_dict(), MODEL_WEIGHTS_PATH) # save model weights in a file
    print('Finished Training')

    # testing
    valid_epoch_acc = validate(
        net,
        testloader
    )
    print(f'Final validation accuracy: {valid_epoch_acc:.3f} %')

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
            _, predictions = torch.max(outputs, 1) # get predictions

            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

def load_model():
    model = resnet18(num_classes=10).cuda()
    model.load_state_dict(torch.load("model_weights_acc_81.pth")) # load model
    model.eval() # set model state as evaluating predictions

    # final training accuracy
    train_epoch_acc = validate(model, trainloader)
    print(f"Final training accuracy: {train_epoch_acc:.3f}%")

    # testing
    valid_epoch_acc = validate(model, testloader)
    print(f'Final validation accuracy: {valid_epoch_acc:.3f} %')

if __name__ == '__main__':
    load_model()