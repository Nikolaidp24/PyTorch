"""
author: Nikolai Peisong Li
"""

import numpy as np
import os
import torch
import torchvision
import matplotlib.pyplot as plt
import torchmetrics

from torch import nn
from torchvision import transforms, datasets
from torchvision.transforms import ToTensor
from torchmetrics import Accuracy, ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# setting train and test directory
train_dir = 'Rice_Image_Dataset/train'
test_dir = 'Rice_Image_Dataset/test'

# setting different transform method of training and testing data
train_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

# using ImageFolder to load in the raw image data
train_data = datasets.ImageFolder(root=train_dir,
                                  transform=train_transform,
                                  target_transform=None)

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=test_transform,
                                 target_transform=None)

# getting the class names
class_names = train_data.classes

# setting batch_size to speed up the training and testing cycle
BATCH_SIZE = 128

# create dataloaders for training and testing datasets using all cpus
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              num_workers=os.cpu_count(),
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             num_workers=os.cpu_count(),
                             shuffle=False)


# Building Model_0 with 2 cnn block and relu activation function and flatten as the last step
class RiceModelV0(nn.Module):
    def __init__(self, input_shape: int, output_shape: int, hidden_units: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 13 * 13,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.block2(self.block1(x)))


# building train_step function
def train_step(model: nn.Module, loss_function: nn.Module, accuracy_function, optim: torch.optim.Optimizer,
               dataloader: torch.utils.data.DataLoader):
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # X, y = X.to('mps'), y.to('mps')
        y_pred = model(X)
        loss = loss_function(y_pred, y)
        train_loss += loss
        train_acc += accuracy_function(y_pred.argmax(dim=1), y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


# building test_step function
def test_step(model: nn.Module, loss_function: nn.Module, accuracy_function, dataloader: torch.utils.data.DataLoader):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            # X, y = X.to('mps'), y.to('mps')
            y_pred = model(X)
            test_loss += loss_function(y_pred, y)
            test_acc += accuracy_function(y_pred.argmax(dim=1), y)
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

    return test_loss, test_acc


# combine training and testing steps, and append the result into a dictionary
def train_model(
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        loss_function: nn.Module,
        optim: torch.optim.Optimizer,
        accuracy_function,
        epochs: int = 5
):
    results = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    for _ in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           loss_function=loss_function,
                                           accuracy_function=accuracy_function,
                                           optim=optim,
                                           dataloader=train_dataloader)

        test_loss, test_acc = test_step(model=model,
                                        loss_function=loss_function,
                                        accuracy_function=accuracy_function,
                                        dataloader=test_dataloader)

        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)

        print(
            f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}')

    return results


# instantiate the model with 3 color channels, 10 hidden layers and outputs 5 classes
rice_model_0 = RiceModelV0(input_shape=3,
                           output_shape=5,
                           hidden_units=10)

# accuracy function using the built-in torchmetrics Accuracy module
acc_fn = Accuracy(task='multiclass', num_classes=len(class_names))

# CrossEntropyLoss because this is a multi-class classification problem
loss_fn = nn.CrossEntropyLoss()

# using Adam optimizer of learning rate=0.001
optimizer = torch.optim.Adam(params=rice_model_0.parameters(),
                             lr=0.001)

# wrap and main script and run, saving the state dict of this model
if __name__ == '__main__':
    run = input('Train model? (y/n): ')
    if run == 'y':
        final = train_model(
            model=rice_model_0,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            loss_function=loss_fn,
            accuracy_function=acc_fn,
            optim=optimizer,
            epochs=3
        )
        torch.save(rice_model_0.state_dict(), 'rice_model_0_state_dict.pt')

    elif run != 'n':
        raise ValueError('Please enter either "y" or "n".')

    else:
        rice_model_loaded = RiceModelV0(input_shape=3,
                                        output_shape=5,
                                        hidden_units=10)
        rice_model_loaded.load_state_dict(torch.load('rice_model_0_state_dict.pt'))
        print('Trained model state dict successfully loaded!')

        y_preds = []
        rice_model_loaded.eval()
        with torch.inference_mode():
            print('Evaluating model...')
            for batch, (X, y) in tqdm(enumerate(test_dataloader)):
                y_logit = rice_model_loaded(X)
                y_pred = torch.softmax(y_logit.squeeze(), dim=1).argmax(dim=1)
                y_preds.append(y_pred)
        y_preds_tensor = torch.cat(y_preds)

        print('Plotting ConfusionMatrix...')
        cm = ConfusionMatrix(task='multiclass', num_classes=len(class_names))
        cm_tensor = cm(preds=y_preds_tensor, target=torch.tensor(test_data.targets))

        plot_confusion_matrix(conf_mat=cm_tensor.numpy(), class_names=class_names)
        plt.show()
