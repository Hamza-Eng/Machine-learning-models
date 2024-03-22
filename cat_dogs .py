import os
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

# Define a transform to normalize the data
transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], 
                                                     [0.5, 0.5, 0.5])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder('mini_project/dataset/train', transform=transform)
# test_data = datasets.ImageFolder('mini_project/dataset/test', transform=transform)

# Using the image datasets and the transforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
# testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
#module ==>
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 500)
        self.fc2 = nn.Linear(500, 2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# instantiate the CNN
model = Net()
print(model)

# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# number of epochs to train the model
n_epochs = 30

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in trainloader:
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        
    # print training statistics 
    train_loss = train_loss/len(trainloader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
########################################################################################
torch.save(model.state_dict(), 'my_model.pth')
####################
import torch
from torchvision import transforms
from PIL import Image

# Define the same transformations used for training
transform = transforms.Compose([
    transforms.Resize(255),  # Might be a typo, common resize is 224
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load the trained model state dictionary
model_state = torch.load('my_model.pth')  # Replace with your model path

# Create a new model instance
model = Net()  # Replace "Net" with your actual model class name

# Load the state dictionary into the model
model.load_state_dict(model_state)

# Set the model to evaluation mode
model.eval()

# Define the image path (replace with your image path)
# image_path = 'C:/Users/hamza/Desktop/9raya& taalim/master FP/python/TD & TP/python/mini_project/dataset/test/Cat/1.jpg'
image_path = 'C:/Users/hamza/Desktop/9raya& taalim/master FP/python/TD & TP/python/mini_project/dataset/train/Dog/33.jpg'

# Load the image
img = Image.open(image_path)

# Preprocess the image
img = transform(img)

# Add a batch dimension for the model (assumes single image prediction)
img = img.unsqueeze(0)

# Predict the class
with torch.no_grad():  # Disable gradient calculation for prediction
    output = model(img)

# Assuming your model has two output units (Cat and Dog)
# Get the predicted class probabilities
predicted_probs = torch.nn.functional.softmax(output.data, dim=1)  # Softmax for probabilities

# Get the class index with the highest probability
_, predicted_index = torch.max(predicted_probs, 1)

# Map the predicted index to class label (assuming 0: Cat, 1: Dog)
class_label = ['Cat', 'Dog'][predicted_index.item()]

# Print the prediction result
print(f"Predicted class: {class_label}")
