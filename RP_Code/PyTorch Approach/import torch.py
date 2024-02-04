import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchview import draw_graph



cinic_dir = 'D:/MDS/2023/4th Tri-3/DLF/Ass2/DS_10283_3192'
traindir = cinic_dir + '/train'
validatedir = cinic_dir + '/valid'
testdir = cinic_dir + '/test'

cinic_mean = [0.47889522, 0.47227842, 0.43047404]
cinic_std = [0.24205776, 0.23828046, 0.25874835]
normalize = transforms.Normalize(mean=cinic_mean, std=cinic_std)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cinic_mean, std=cinic_std)
])

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=cinic_mean, std=cinic_std)
])

trainset = datasets.ImageFolder(root=traindir, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=64,
                                          shuffle=True,
                                          num_workers=2)

validateset = datasets.ImageFolder(root=validatedir, transform=transform)
validateloader = torch.utils.data.DataLoader(validateset,
                                             batch_size=64,
                                             shuffle=True,
                                             num_workers=2)

testset = datasets.ImageFolder(root=testdir, transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=64,
                                         shuffle=True,
                                         num_workers=2)

classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# select the first 4 images
images = images[:4]
labels = labels[:4]

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)

model_graph = draw_graph(net, input_size=(3,32,32), expand_nested=True)
model_graph.visual_graph
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
# torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
epoch_num = 2
def train_model_one_epoch(model, trainloader, optimizer, criterion, device):
    model.train()
    
    result = {'loss': 0,
             'accuracy': 0,
             'recall': 0,
             'precision': 0}
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        result['loss'] += loss.item()
        result['accuracy'] += accuracy(outputs, labels)
        result['recall'] += recall(outputs, labels)
        result['precision'] += precision(outputs, labels)

        loss.backward()
        optimizer.step()
        
    result = {k: v / len(trainloader) for k, v in result.items()}
    return result 

def valid_model(model, validloader, criterion, device):
    result = {'loss': 0,
             'accuracy': 0,
             'recall': 0,
             'precision': 0}
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(validloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            result['loss'] += loss.item()
            result['accuracy'] += accuracy(outputs, labels)
            result['recall'] += recall(outputs, labels)
            result['precision'] += precision(outputs, labels)

    result = {k: v / len(validloader) for k, v in result.items()}
    return result

def train_model(model, trainloader, validloader, optimizer, criterion, num_epochs, device):
    best_valid_perform = None
    for epoch in range(num_epochs):
        train_perform = train_model_one_epoch(model, trainloader, optimizer, criterion, device)
        valid_perform = valid_model(model, validloader, criterion, device)
        if cmp_perform(best_valid_perform, valid_perform):
            assign_perform(best_valid_perform, valid_perform)
            torch.save(model.state_dict(), 'best_valid.pt')
        print(f'Epoch {epoch+1}: Train Loss={train_perform["loss"]:.4f}, Train Acc={train_perform["accuracy"]:.4f}, Valid Loss={valid_perform["loss"]:.4f}, Valid Acc={valid_perform["accuracy"]:.4f}')

    return train_perform, valid_perform
        
for epoch in range(epoch_num):
    train_perform = train_model_one_epoch(net, trainloader, optimizer, criterion, device)
    valid_perform = valid_model(net, validateloader, criterion, device)
    if cmp_perform(best_valid_perform, valid_perform):
        assign_perform(best_valid_perform, valid_perform)
        torch.save(net.state_dict(), 'best_valid.pt')
    print(f'Epoch {epoch + 1}: Train Loss={train_perform["loss"]:.4f}, Train Acc={train_perform["accuracy"]:.4f}, Valid Loss={valid_perform["loss"]:.4f}, Valid Acc={valid_perform["accuracy"]:.4f}')

print('Finished Training')

PATH = './cinic_net.pth'
torch.save(net.state_dict(), PATH)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))