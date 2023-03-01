"""
author: Nikolai Peisong Li
"""

import os
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torchmetrics
import pandas as pd
import torchvision.models as models

from typing import List, Dict, Tuple, Final
from torch import nn
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchmetrics import Accuracy, ConfusionMatrix
from torch.utils.data import DataLoader
from mlxtend.plotting import plot_confusion_matrix
from tqdm.auto import tqdm
from torchvision.models import ResNet18_Weights

# setting up the training and testing directory
train_dir = 'raw-img/train'
test_dir = 'raw-img/test'

# set up train and test transforms, using TrivialAugmentWide to transform train data for more accuracy
train_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor()
])

# set up train and test data using ImageFolder
train_data = datasets.ImageFolder(
    root=train_dir,
    transform=train_transform,
    target_transform=None
)

test_data = datasets.ImageFolder(
    root=test_dir,
    transform=test_transform,
    target_transform=None
)


# make a function to preview an image
def preview_image(data: Tuple):
    plt.imshow(data[0].permute(1, 2, 0))
    #     plt.axis(False)
    plt.xlabel(class_names[data[1]])
    plt.ylabel(data[0].shape)
    plt.show()


preview_image(train_data[0])

# set up dataloader
BATCH_SIZE: Final = 64

train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=True,
    pin_memory=True
)

test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=False,
    pin_memory=True
)


class AnimalsBaselineModel(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_units):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=2,
                padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=2,
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
                stride=2,
                padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=2,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_units * 3 * 3,
                out_features=output_shape
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.block2(self.block1(x)))


device = 'mps' if torch.backends.mps.is_available() else 'cpu'

model_0 = AnimalsBaselineModel(input_shape=3, output_shape=10, hidden_units=50).to(device)


def train_step(
        model: nn.Module,
        loss_function: nn.Module,
        accuracy_function,
        optim: torch.optim.Optimizer,
        dataloader: torch.utils.data.DataLoader
):
    train_loss, train_acc = 0, 0
    model.train()
    print('Starting training cycle:\n------------')

    for X, y in tqdm(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_function(y_pred, y)
        train_loss += loss.item()
        train_acc += accuracy_function(y_pred.argmax(dim=1), y).item()
        optim.zero_grad()
        loss.backward()
        optim.step()

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc * 100


def test_step(
        model: nn.Module,
        loss_function: nn.Module,
        accuracy_function,
        dataloader: torch.utils.data.DataLoader
):
    test_loss, test_acc = 0, 0
    model.eval()
    print('Starting testing cycle:\n--------------')
    with torch.inference_mode():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            test_loss += loss_function(y_pred, y).item()
            test_acc += accuracy_function(y_pred.argmax(dim=1), y).item()

        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)

    return test_loss, test_acc * 100


def train_model(
        model: nn.Module,
        loss_function: nn.Module,
        optim: torch.optim.Optimizer,
        accuracy_function,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        epochs: int = 5
) -> Dict:
    results = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    for _ in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           loss_function=loss_function,
                                           optim=optim,
                                           accuracy_function=accuracy_function,
                                           dataloader=train_loader)

        test_loss, test_acc = test_step(model=model,
                                        loss_function=loss_function,
                                        accuracy_function=accuracy_function,
                                        dataloader=test_loader)

        print(
            f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')

        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)

    return results


# set up optimizer, loss function and accuracy function
optimizer = torch.optim.Adam(params=model_0.parameters(),
                             lr=0.001)
loss_fn = nn.CrossEntropyLoss()
acc_fn = Accuracy(task='multiclass', num_classes=len(class_names)).to(device)

model_0_result = train_model(model=model_0,
                             loss_function=loss_fn,
                             accuracy_function=acc_fn,
                             optim=optimizer,
                             train_loader=train_dataloader,
                             test_loader=test_dataloader,
                             epochs=10)


def plot_curves_single(result: Dict[str, List[float]]):
    trainloss = result['train_loss']
    trainacc = result['train_acc']
    testloss = result['test_loss']
    testacc = result['test_acc']

    epochs = range(len(result['train_loss']))

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, trainloss, label='Train Loss')
    plt.plot(epochs, testloss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.title('Loss Curve')
    plt.ylim(0, 2.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, trainacc, label='Train Acc')
    plt.plot(epochs, testacc, label='Test Acc')
    plt.xlabel('Epoch')
    plt.title('Acc Curve')
    plt.ylim((0, 100))
    plt.legend()

    plt.show()


def dict_to_df(raw_dict: List):
    converted = []
    for index, dic in enumerate(raw_dict):
        convert = pd.DataFrame(dic)
        convert['model_name'] = f'model_{index}'
        converted.append(convert)
    df_combined = pd.concat(converted, ignore_index=True, axis=0)
    return df_combined


def plot_curves(*result):
    results = list(result)
    if len(results) == 0:
        raise ValueError(
            'Expected at least one result to be able to plot!'
        )
    if len(results) == 1:
        plot_curves_single(results[0])
        return
    df = dict_to_df(results)
    col_names = list(df.iloc[:, :4].columns)
    length = []
    for ls in list(df['model_name'].unique()):
        length.append(len(df.loc[df['model_name'] == ls]))
    epochs = range(max(length))
    model_names = list(df['model_name'].unique())
    for i, col in enumerate(col_names):
        plt.subplot(2, 2, i + 1)
        for j, model in enumerate(model_names):
            plt.plot(epochs, np.array(df.loc[df.model_name == model, col]), label=model)
        plt.xlabel('Epoch')
        plt.title(f'{col.split("_")[1]} curve')
        if col == 'train_loss' or col == 'test_loss':
            plt.ylim((0, 2.5))
        elif col == 'train_acc' or col == 'test_acc':
            plt.ylim((0, 100))
        plt.legend()
        plt.show()


plot_curves(model_0_result)


# making an evaluate function
def eval_model(
        model: nn.Module,
        loss_function: nn.Module,
        accuracy_function,
        dataloader: torch.utils.data.DataLoader
):
    loss, acc = 0, 0
    model.eval()
    print(f'Evaluating...\n------------')
    with torch.inference_mode():
        for batch, (X, y) in enumerate(tqdm(dataloader)):
            X, y = X.to(device), y.to(device)
            y_logit = model(X)
            y_pred = torch.softmax(y_logit.squeeze(), dim=1).argmax(dim=1)
            loss += loss_function(y_logit, y).item()
            acc += accuracy_function(y_pred, y).item()

        loss /= len(dataloader)
        acc /= len(dataloader)
    return {
        'Model Name': model.__class__.__name__,
        'Loss': loss,
        'Accuracy': acc * 100
    }


model_0_eval = eval_model(model=model_0, loss_function=loss_fn, accuracy_function=acc_fn, dataloader=test_dataloader)

# saving the baseline model's state dict
torch.save(model_0.state_dict(), 'BaseLineModel.pt')


class AnimalsModelV1(nn.Module):
    def __init__(self, input_shape: int, output_shape: int, hidden_units: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=2,
                padding=0
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=2,
                padding=0
            ),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=2,
                padding=0
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=2,
                padding=0
            ),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 3 * 3,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.block2(self.block1(x)))


model_1 = AnimalsModelV1(input_shape=3, output_shape=10, hidden_units=100).to(device)

optimizer_1 = torch.optim.Adam(params=model_1.parameters(),
                               lr=0.001)

model_1_result = train_model(model=model_1,
                             loss_function=loss_fn,
                             accuracy_function=acc_fn,
                             optim=optimizer_1,
                             train_loader=train_dataloader,
                             test_loader=test_dataloader,
                             epochs=10)

plot_curves(model_0_result, model_1_result)

model_1_eval = eval_model(model=model_1,
                          loss_function=loss_fn,
                          accuracy_function=acc_fn,
                          dataloader=test_dataloader)

# comparison of the baseline and model_1
plot_curves(model_0_result, model_1_result)

# add augmentation to the transform
train_transform_trivial = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

train_data_1 = datasets.ImageFolder(
    root=train_dir,
    transform=train_transform_trivial,
    target_transform=None
)

train_dataloader_1 = DataLoader(
    dataset=train_data_1,
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=True,
    pin_memory=True
)

model_2 = AnimalsModelV1(input_shape=3, output_shape=len(class_names), hidden_units=100).to(device)

optimizer_2 = torch.optim.Adam(params=model_2.parameters(),
                               lr=0.001)

model_2_result = train_model(
    model=model_2,
    loss_function=loss_fn,
    accuracy_function=acc_fn,
    train_loader=train_dataloader_1,
    test_loader=test_dataloader,
    optim=optimizer_2,
    epochs=10
)

model_2_eval = eval_model(model=model_2,
                          dataloader=test_dataloader,
                          loss_function=loss_fn,
                          accuracy_function=acc_fn)

plot_curves(model_0_result, model_1_result, model_2_result)

# add more layers and lower the learning rate further
model_3 = AnimalsModelV1(input_shape=3, output_shape=len(class_names), hidden_units=200).to(device)
optimizer_3 = torch.optim.Adam(params=model_3.parameters(),
                               lr=0.001)

model_3_result = train_model(
    model=model_3,
    train_loader=train_dataloader_1,
    test_loader=test_dataloader,
    loss_function=loss_fn,
    accuracy_function=acc_fn,
    optim=optimizer_3,
    epochs=10
)

model_3_eval = eval_model(model=model_3,
                          dataloader=test_dataloader,
                          accuracy_function=acc_fn,
                          loss_function=loss_fn)

plot_curves(model_0_result, model_1_result, model_2_result, model_3_result)

torch.save(model_1.state_dict(), 'Model_1.pt')
torch.save(model_2.state_dict(), 'Model_2.pt')
torch.save(model_3.state_dict(), 'Model_3.pt')

weights = ResNet18_Weights.DEFAULT
model_4 = models.resnet18(weights=weights).to(device)
auto_transform = weights.transforms()
train_data_transfer = datasets.ImageFolder(
    root=train_dir,
    transform=auto_transform,
    target_transform=None
)

test_data_transfer = datasets.ImageFolder(
    root=test_dir,
    transform=auto_transform,
    target_transform=None
)

train_dataloader_transfer = DataLoader(
    dataset=train_data_transfer,
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=True,
    pin_memory=True
)

test_dataloader_transfer = DataLoader(
    dataset=test_data_transfer,
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=False,
    pin_memory=True
)

for param in model_4.parameters():
    param.requires_grad = False

model_4.fc = nn.Linear(in_features=512, out_features=len(class_names), bias=True).to(device)

optimizer_resnet18 = torch.optim.Adam(params=model_4.parameters(),
                                      lr=0.001)

model_4_result = train_model(
    model=model_4,
    loss_function=loss_fn,
    accuracy_function=acc_fn,
    optim=optimizer_resnet18,
    train_loader=train_dataloader_transfer,
    test_loader=test_dataloader_transfer,
    epochs=10
)

model_4_eval = eval_model(model=model_4,
                          loss_function=loss_fn,
                          accuracy_function=acc_fn,
                          dataloader=test_dataloader_transfer)

y_preds = []
model_4.eval()
with torch.inference_mode():
    for X, y in tqdm(test_dataloader_transfer):
        X, y = X.to(device), y.to(device)
        y_logit = model_4(X)
        y_pred = torch.softmax(y_logit.squeeze(0), dim=1).argmax(dim=1)
        y_preds.append(y_pred.cpu())
y_preds_tensor = torch.cat(y_preds)

cm = ConfusionMatrix(task='multiclass', num_classes=len(class_names))
cm_tensor = cm(preds=y_preds_tensor, target=torch.tensor(test_data_transfer.targets))
plot_confusion_matrix(conf_mat=cm_tensor.numpy(), class_names=class_names)
plt.show()

models_eval = [model_0_eval, model_1_eval, model_2_eval, model_3_eval, model_4_eval]

dfs = []
for model in models_eval:
    df = pd.DataFrame(model, index=[0])
    dfs.append(df)
df_models_eval = pd.concat(dfs, ignore_index=True, axis=0)

df_models_eval.iloc[0, 0] = 'Baseline'
df_models_eval.iloc[1, 0] = 'Model_1'
df_models_eval.iloc[2, 0] = 'Model_2'
df_models_eval.iloc[3, 0] = 'Model_3'

df_models_eval[['Model Name', 'Loss']].plot(kind='bar', color='orange')
plt.xticks(ticks=[0, 1, 2, 3, 4] ,labels=list(df_models_eval['Model Name']), rotation=0)
df_models_eval[['Model Name', 'Accuracy']].plot(kind='bar')
plt.xticks(ticks=[0, 1, 2, 3, 4] ,labels=list(df_models_eval['Model Name']), rotation=0)
plt.show()

torch.save(model_4.state_dict(), 'resnet18.pt')

