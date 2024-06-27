import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
import torch.nn as nn
from torchvision.models import resnet18
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

MODEL_WEIGHTS_PATH = "model_weights.pth"
DEVICE = torch.device("cuda")

# load CIFAR-10
transform_train = transforms.Compose([
    transforms.Resize((32, 32)),  # resizes the image to 32x32
    transforms.RandomHorizontalFlip(), # flips the image w.r.t horizontal axis
    transforms.RandomRotation(10), # rotates the image to a random angle
    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), # performs actions like zooms, change shear angles, etc.
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize image between [-1, 1]
])

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize image between [-1, 1]
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

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

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

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # calculate outputs by running images through the network
            outputs = model(images)

            # the class with the highest energy is what we choose as prediction
            predicted = torch.argmax(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy

def find_accuracies_of_each_class(net):
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = net(images)
            predictions = torch.argmax(outputs, 1) # predictions

            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    
    return total_pred, correct_pred

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
        # update learning rate at epoch 20 to 10x reduction from initial
        if epoch == 20:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001
        # update learning rate at epoch 50 to 10x from at epoch 20
        if epoch == 50:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.00001
        
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

    # print accuracy for each class
    total_pred, correct_pred = find_accuracies_of_each_class(net)
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


# load a model
def load_model(testloader):
    model = resnet18(num_classes=10).cuda()
    model.load_state_dict(torch.load("model_weights_acc_81.pth")) # load model
    model.eval() # set model state as evaluating predictions

    # final training accuracy
    # train_epoch_acc = validate(model, trainloader)
    # print(f"Final training accuracy: {train_epoch_acc:.3f}%")

    # validation
    # valid_epoch_acc = validate(model, testloader)
    # print(f'Final validation accuracy: {valid_epoch_acc:.3f} %')

    # print accuracy for each class
    # total_pred, correct_pred = find_accuracies_of_each_class(model)
    # for classname, correct_count in correct_pred.items():
    #     accuracy = 100 * float(correct_count) / total_pred[classname]
    #     print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    # loss function, optimizer (for testing adversarial stuff)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # check accuracy of model using adversarial images with 1 epsilon applied over entire image
    # for epsilon in [0.25, 1.0, 1.5]:
    #     valid_epoch_acc = validate_with_adversarial_examples(
    #         model,
    #         testloader,
    #         optimizer,
    #         criterion,
    #         epsilon
    #     )
    #     print(f'Validation accuracy (adversarials) with epsilon {epsilon}: {valid_epoch_acc:.3f} %')

    # check accuracy of model using adversarial images with 2/255 epsilon applied on middle 16x16 and 8/255 applied on the rest
    valid_epoch_acc = validate_with_adversarials_using_two_epsilons(
        model,
        testloader,
        optimizer,
        criterion,
        8.0/255,
        2.0/255
    )
    print(f'Validation accuracy with adversarials: {valid_epoch_acc:.3f} %')

# generates adversarial image from given image, epsilon and image data gradient
def generate_adversarial(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign() # get sign of gradient
    adversarial = image + epsilon * sign_data_grad
    adversarial = torch.clamp(adversarial, 0, 1) # clamp between 0 and 1
    return adversarial

# denormalizes a batch of images
def denorm(batch, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    mean = torch.tensor(mean).to(DEVICE)
    std = torch.tensor(std).to(DEVICE)
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

# generates adversarial image with 1 epsilon applied in middle 16x16 and another epsilon applied everywhere else
def generate_adversarial_two_epsilons(image, eps1, eps2, data_grad):
    sign_data_grad = data_grad.sign()
    adversarial = image + eps1 * sign_data_grad # apply eps1 on entire image
    adversarial[:, :, 8:24, 8:24] = image[:, :, 8:24, 8:24] + eps2 * sign_data_grad[:, :, 8:24, 8:24] # slice middle
    adversarial = torch.clamp(adversarial, 0, 1) # clamp in [0, 1]
    return adversarial

# check accuracy with one epsilon applied on middle 16x16 and another epsilon applied on the rest
def validate_with_adversarials_using_two_epsilons(model, testloader, optimizer, criterion, overall_eps, center_eps):
    model.train()
    correct = 0
    for _, data in enumerate(testloader, 0):
        # STEP 1: Train the model on the original image to obtain gradient
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        inputs.requires_grad = True
        # zero the parameter gradients
        optimizer.zero_grad()
        # get outputs
        outputs = model(inputs)
        # compute loss, go backwards
        loss = criterion(outputs, labels)
        # backpropagation
        loss.backward()
        # commented out because we don't actually want to update weights in the model; we just want to get the gradients
        # optimizer.step()

        # STEP 2: Generate adversarial image of whole
        # get gradients of image
        img_grad = inputs.grad.data
        # unnormalize to [0, 1]
        images_denorm = denorm(inputs)
        # generate adversarial
        adversarials = generate_adversarial_two_epsilons(images_denorm, overall_eps, center_eps, img_grad)
        # re-normalize back to [-1, 1]
        adversarials_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(adversarials)

        # predict on model
        adv_outputs = model(adversarials_norm)
        # get number of correct
        correct += (torch.argmax(adv_outputs, 1) == labels).sum().item()
    
    accuracy = 100.0 * correct / len(trainset)
    return accuracy

# def validate_with_adversarials_using_two_epsilons2(model, testloader, optimizer, criterion, overall_eps, center_eps):
#     model.train()
#     correct = 0
#     for _, data in enumerate(testloader, 0):
#         # STEP 1: Train the model on the original image to obtain gradient
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data
#         inputs = inputs.to(DEVICE)
#         labels = labels.to(DEVICE)
#         inputs.requires_grad = True
#         # zero the parameter gradients
#         optimizer.zero_grad()
#         # get outputs
#         outputs = model(inputs)
#         # compute loss, go backwards
#         loss = criterion(outputs, labels)
#         # backpropagation
#         loss.backward()
#         # commented out because we don't actually want to update weights in the model; we just want to get the gradients
#         # optimizer.step()

#         # STEP 2: Generate adversarial image of whole
#         # get gradients of image
#         img_grad = inputs.grad.data
#         # unnormalize to [0, 1]
#         images_denorm = denorm(inputs)
#         # generate adversarial
#         adversarials = generate_adversarial(images_denorm, overall_eps, img_grad)
#         # re-normalize back to [-1, 1]
#         adversarials_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(adversarials)

#         # STEP 3: Generate adversarial image of center 16x16
#         # crop center 16x16
#         images_center = v2.CenterCrop(size=16)(inputs)
#         # crop center 16x16 of image gradient
#         img_grad_center = v2.CenterCrop(size=16)(img_grad)
#         # unnormalize to [0, 1]
#         img_center_denorm = denorm(images_center)
#         # generate adversarial
#         adversarials_center = generate_adversarial(img_center_denorm, center_eps, img_grad_center)
#         # re-normalize
#         adversarials_center_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(adversarials_center)

#         # plot_examples(inputs)
#         # plot_examples(adversarials_norm)
#         # plot_examples(images_center)

#         # STEP 4: Re-copy 16x16
#         for k1 in range(4): # copy for all 4 tensors per batch
#             for k2 in range(3): # copy over all 3 color channels per image
#                 for i in range(16):
#                     for j in range(16):
#                         adversarials_norm[k1][k2][i + 8][j + 8] = adversarials_center_norm[k1][k2][i][j]

#         # predict on model
#         adv_outputs = model(adversarials_norm)
#         # get number of correct
#         correct += (torch.argmax(adv_outputs, 1) == labels).sum().item()

#         # plot_examples(adversarials_center_norm)
#         # plot_examples(adversarials_norm)
    
#     accuracy = 100.0 * correct / len(trainset)
#     return accuracy

# check accuracy with 1 epsilon applied across entire image
def validate_with_adversarial_examples(model, testloader, optimizer, criterion, epsilon):
    model.train()
    correct = 0
    for _, data in enumerate(testloader, 0):
        # STEP 1: Train the model on the original image to obtain gradient
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        inputs.requires_grad = True
        # zero the parameter gradients
        optimizer.zero_grad()
        # get outputs
        outputs = model(inputs)
        # compute loss, go backwards
        loss = criterion(outputs, labels)
        # backpropagation
        loss.backward()
        # commented out because we don't actually want to update weights in the model; we just want to get the gradients
        # optimizer.step()

        # STEP 2: Generate adversarial example from original images and gradients, then test model on those
        # get gradients of image
        img_grad = inputs.grad.data
        # unnormalize to [0, 1]
        images_denorm = denorm(inputs)
        # generate adversarial
        adversarials = generate_adversarial(images_denorm, epsilon, img_grad)
        # re-normalize back to [-1, 1]
        adversarials_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(adversarials)
        # predict on model
        adv_outputs = model(adversarials_norm)
        # get number of correct
        correct += (torch.argmax(adv_outputs, 1) == labels).sum().item()

        # plot_examples(inputs.to(torch.device("cpu")))
        # plot_examples(adversarials_norm.to(torch.device("cpu")))
    
    accuracy = 100.0 * correct / len(trainset)
    return accuracy

# # show images
# def imshow(img):
#     img = img / 2 + 0.5 # unnormalize from [-1, 1] to [0, 1]
#     npimg = img.numpy()
#     displayed = np.transpose(npimg, (1, 2, 0))
#     plt.imshow(displayed)
#     # plt.imshow(displayed @ [0.299, 0.587, 0.113], cmap='gray') # grayscale display
#     plt.show()

# # plot grid of images
# def plot_examples(images):
#     images = images.to(torch.device("cpu"))
#     imshow(torchvision.utils.make_grid(images))

if __name__ == '__main__':
    load_model(testloader)
    # model()