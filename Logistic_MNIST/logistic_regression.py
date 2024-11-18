# **1. Importing Required Libraries**

import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transform
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import SubsetRandomSampler, DataLoader

# **2. Loading MNIST Dataset**

# Download and load the MNIST dataset with the necessary transformation
dataset = MNIST(root='./data', download=True, train=True, transform=transform.ToTensor())
testset = MNIST(root='./data', download=True, train=False, transform=transform.ToTensor())

# Setting up the device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# **3. Visualizing an Example Image**

# Fetch a sample image and label from the dataset
images, labels = dataset[230]

# Crop the image for better focus
plot = images[:, 10:17, 10:17]

# Display the cropped image using matplotlib
plt.imshow(plot[0], cmap='gray')
plt.show()

# Output the label of the image
print("Labels: ", labels)

# **4. Splitting Data into Training and Validation Sets**

def splitted_data(data, valid_data_percent):
    """
    Splits the dataset into training and validation sets.
    """
    valid_num = int(data * valid_data_percent)  # Calculate the number of validation samples
    index = np.random.permutation(data)  # Shuffle the data indices
    return index[valid_num:], index[:valid_num]  # Return training and validation sets

# Split the data with 25% validation data
training_data, validation_data = splitted_data(len(dataset), 0.25)

# Output the sizes of the training and validation datasets
print("Training data size: ", len(training_data))
print("Validation data size: ", len(validation_data))

# **5. Creating Data Loaders**

# Creating samplers for the training and validation sets
training_sampler = SubsetRandomSampler(training_data)
validation_sampler = SubsetRandomSampler(validation_data)

# Set batch size for loading data
batch_size = 100

# Create DataLoader instances for both training and validation datasets
training_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=training_sampler)
validation_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=validation_sampler)

# **6. Defining the Model Architecture**

input_size = 28 * 28  # Input size (28x28 pixels)
hidden_size1 = 256     # First hidden layer size
hidden_size2 = 128     # Second hidden layer size
output_size = 10       # Output size (10 digits)

class MNISTMODEL(nn.Module):
    """
    Defines the architecture of the neural network for MNIST classification.
    """
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        # Define the layers
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)

    def forward(self, batch):
        """
        Forward pass through the network.
        """
        # Flatten the input batch of images
        size = batch.view(batch.size(0), -1)

        # Pass through the first hidden layer with ReLU activation
        hidden1 = self.linear1(size)
        output = F.relu(hidden1)

        # Pass through the second hidden layer with ReLU activation
        hidden2 = self.linear2(output)
        output = F.relu(hidden2)

        # Output layer
        output = self.linear3(output)

        return output

# Instantiate the model and move it to the selected device
model = MNISTMODEL(input_size, hidden_size1, hidden_size2, output_size).to(device)

# **7. Checking Model Parameters**

# Print the model parameters to understand its structure
for t in model.parameters():
    print(t.shape)
    print(t.device)

# **8. Forward Pass & Predictions**

# Make a forward pass using the training data
for images, labels in training_loader:
    images, labels = images.to(device), labels.to(device)
    prediction = model(images)

# **9. Softmax Output**

# Display the raw output of the model for the first image
print(prediction[0])

# Apply Softmax to convert the output to probabilities
prob_sum = torch.sum(prediction[0])
print(prob_sum)

# Apply Softmax on the output
changed_pred = F.softmax(prediction, dim=1)
changed_sum = torch.sum(changed_pred[9])
print(changed_sum)

# **10. Maximum Value Prediction**

# Get the predicted class with the highest probability
_, pred = torch.max(prediction, dim=1)
print(pred)

# **11. Display True Labels**

# Show the true labels for the current batch
print(labels)

# **12. Accuracy Calculation**

def accuracy(output, labels):
    """
    Computes the accuracy of predictions.
    """
    _, pred = torch.max(output, dim=1)
    return torch.sum(pred == labels).item() / len(pred) * 100

# **13. Loss Function**

# Define the loss function (CrossEntropyLoss) for multi-class classification
loss_function = F.cross_entropy

# **14. Optimizer Setup**

# Use Stochastic Gradient Descent (SGD) with learning rate of 0.01
opt = torch.optim.SGD(model.parameters(), lr=0.01)

# **15. Training Loop (Loss & Metrics)**

def loss_batch(model, loss_function, images, labels, opt, metrics=accuracy):
    """
    Computes loss and metrics for a batch of data.
    """
    # Perform forward pass
    prediction = model(images.to(device))

    # Compute the loss
    loss = loss_function(prediction, labels.to(device))

    # Backpropagate and update parameters if optimizer is provided
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    # Compute the accuracy if metrics is provided
    metric_result = None
    if metrics is not None:
        metric_result = metrics(prediction, labels.to(device))

    return loss.item(), len(images), metric_result

# **16. Evaluation Function**

def evaluate(model, loss_function, validation_loader, metrics=accuracy):
    """
    Evaluates the model on the validation set.
    """
    with torch.no_grad():
        result = [loss_batch(model, loss_function, images.to(device), labels.to(device), opt=None, metrics=accuracy)
                  for images, labels in validation_loader]

        losses, num, metric = zip(*result)

        # Calculate total number of samples
        total = np.sum(num)

        # Compute weighted average of loss
        loss = np.sum(np.multiply(losses, num)) / total

        # Compute weighted average of accuracy
        metric = np.sum(np.multiply(metric, num)) / total if metrics is not None else None

    return loss, total, metric

# **17. Full Training Function**

def train(nepochs, model, loss_function, training_loader, validation_loader, opt, metrics=accuracy):
    """
    Trains the model for a given number of epochs.
    """
    for epoch in range(nepochs):
        model.train()
        for images, labels in training_loader:
            images, labels = images.to(device), labels.to(device)
            train_loss, _, train_acc = loss_batch(model, loss_function, images, labels, opt, metrics=accuracy)

        # Evaluate the model on the validation set
        model.eval()
        valid_loss, _, valid_acc = evaluate(model, loss_function, validation_loader, metrics=accuracy)

        # Print the training and validation metrics for each epoch
        print(f"Epoch: {epoch+1}/{nepochs}")
        print(f"Training loss: {train_loss:.4f}, Validation loss: {valid_loss:.4f}")
        print(f"Training accuracy: {train_acc:.2f}%, Validation accuracy: {valid_acc:.2f}%")
        print("---------------------------------------------------------")

    return train_loss, train_acc, valid_loss, valid_acc

# **18. Testing the Model**

def prediction(images, model):
    """
    Predict the class of a single image.
    """
    input = images.to(device).unsqueeze(0)
    output = model(input)
    _, pred = torch.max(output, dim=1)
    return pred[0].item()

# Test the model by predicting a random image from the test set
images, labels = testset[6]
plt.imshow(images[0], cmap='gray')
plt.show()
print("Labels: ", labels)
print("Predicted: ", prediction(images.to(device), model))

# **19. Model Evaluation on Test Set**

test_loader = DataLoader(testset, batch_size=200)

# Evaluate the model on the test dataset
print(evaluate(model, loss_function, test_loader, metrics=accuracy))

# **20. Saving the Model**

# Save the trained model parameters
torch.save(model.state_dict(), "MNIST.pth")

# Loading the saved model
model.load_state_dict(torch.load("MNIST.pth"))
