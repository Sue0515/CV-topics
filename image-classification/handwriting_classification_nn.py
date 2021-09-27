import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import matplotlib.pylab as plt

torch.manual_seed(3) # increases the probability of a different result 
train_data = datasets.MNIST(root='./data', train = True, download = True, transform = transforms.ToTensor())
validation_data = datasets.MNIST(root='./data', train=False, download=True, transform = transforms.ToTensor())

class NeuralNet(nn.Module):
    def __init__(self, input, h1, h2, output):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input, h1)
        self.layer2 = nn.Linear(h1, h2)
        self.layer3 = nn.Linear(h2, output)
    
    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        x = self.layer3(x)
        return x

def train(model, train_loader, validation_loader, loss_function, optimizer, epochs = 100):
    i = 0
    result = {'training_loss':[], 'validation_accuracy':[]}
    #print("train_loader: ", train_loader)
    for epoch in range(epochs):
        correct_prediction = 0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            #print("x.view: ", x.view(-1, 28*28))
            z = model(x.view(-1, 28 * 28))
            #print("z: ", z)
            loss = loss_function(z, y)
            loss.backward()
            optimizer.step()
            result['training_loss'].append(loss.data.item())
        
        for x, y in validation_loader:
            z = model(x.view(-1, 28 * 28))
            _, label = torch.max(z, 1)
            correct_prediction += (y == label).sum().item()

        accuracy = (correct_prediction / len(validation_data))
        result['validation_accuracy'].append(accuracy)
    
    return result 

# set parameters 
input_layer = 28 * 28
hidden_first = 50
hidden_second = 50
output = 10 
epoch = 10
learning_rate = 0.01

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=2000, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_data, batch_size=5000, shuffle=False)
model = NeuralNet(input_layer, hidden_first, hidden_second, output)
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
loss_function = nn.CrossEntropyLoss()

training_result = train(model, train_loader, validation_loader, loss_function, optimizer, epochs = epoch)

plt.plot(training_result['training_loss'], label='sigmoid')
plt.title('Training Loss')
plt.ylabel('loss')

plt.plot(training_result['validation_accuracy'], label = 'sigmoid')
plt.title('Validation Accuracy')
plt.ylabel('accuracy')

