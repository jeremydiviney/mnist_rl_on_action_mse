import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchvision import datasets, transforms, models


from tensorboardX import SummaryWriter
from loaders.MNISTMemLoader import get_loaders
from helpers import train_experiment

print("PyTorch version:", torch.__version__)

print(torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
# Details for GPU index 0
print("GPU details:", torch.cuda.get_device_properties(0))

print(torch.cuda.get_device_name(0))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())


class DenseLinearModel(nn.Module):
    def __init__(self):
        super(DenseLinearModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x


# TODO: try pure random selection instead o f multinomial
# TODO: try max selection
# TODO: try amping the penalty for bad predictions even more

class PolicyLoss(nn.Module):
    def __init__(self, calc_type, max_prob):
        self.calc_type = calc_type
        self.max_prob = max_prob
        super(PolicyLoss, self).__init__()

    def forward(self, predictions, labels):
        # Ensuring the predictions and labels have the same shape
        # assert predictions.shape == labels.shape, "Predictions and labels must have the same shape"

        device = predictions.device
        epsilon = 1e-8

        final_predictions = torch.softmax(predictions, dim=1)

        _, action = final_predictions.max(dim=1)
        action = torch.multinomial(final_predictions, 1).squeeze()

        reward = torch.where(action == labels, 1, -1)
        chosen_action_probs = final_predictions[range(final_predictions.size(0)), action]

        if self.calc_type == 0:
            good_loss = -torch.log(chosen_action_probs[reward == 1])
            bad_loss = -torch.log(1 - chosen_action_probs[reward == -1])
            loss = torch.concatenate((good_loss, bad_loss))

        elif self.calc_type == 1:
            good_loss = (self.max_prob - chosen_action_probs[reward == 1])**2
            bad_loss = (chosen_action_probs[reward == -1])**2
            loss = torch.concatenate((good_loss, bad_loss))

        elif self.calc_type == 2:
            good_loss = (1 - chosen_action_probs[reward == 1]) * 1
            bad_loss = chosen_action_probs[reward == -1] * 1
            loss = torch.concatenate((good_loss, bad_loss))

        elif self.calc_type == 3:
            good_loss = (1 - chosen_action_probs[reward == 1]) * 0
            bad_loss = chosen_action_probs[reward == -1] * 1
            loss = torch.concatenate((good_loss, bad_loss))

        elif self.calc_type == 4:
            good_loss = (1 - chosen_action_probs[reward == 1]) * 1
            bad_loss = chosen_action_probs[reward == -1] * 0
            loss = torch.concatenate((good_loss, bad_loss))

        return loss.mean()


def get_model():
    return DenseLinearModel()


def get_model_resnet():
    resnet18 = models.resnet18(pretrained=False)
    resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 10)

    return resnet18


train_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,)),
                                     transforms.RandomRotation(degrees=(-8.5, 8.5))]
                                     )

val_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))]
                                   )

train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True,
                                           transform=train_transform), batch_size=128, shuffle=True, num_workers=6)
val_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, download=True,
                                         transform=val_transform), batch_size=1024, shuffle=False, num_workers=6)


# Define your experiments
experiments = [
    # {
    #     'name': 'MNIST Baseline (Adam, StepLR)',
    #     'model_fn': get_model_resnet,
    #     'model_params': {},
    #     'train_loader': train_loader,
    #     'val_loader': val_loader,
    #     'num_epochs': 20,
    #     'scheduler_class': StepLR,
    #     'scheduler_params': {'step_size': 2, 'gamma': 0.75},
    #     'optimizer_class': optim.Adam,
    #     'optimizer_params': {'lr': .002},
    #     'loss_fn': nn.CrossEntropyLoss(),
    #     'num_runs': 1,
    #     'train_fn': train_experiment
    # },

    {
        'name': 'MNIST RL Policy (Adam, StepLR) RESNET18 calc_type=0',
        'model_fn': get_model_resnet,
        'model_params': {},
        'train_loader': train_loader,
        'val_loader': val_loader,
        'num_epochs': 20,
        'scheduler_class': ReduceLROnPlateau,
        # 'scheduler_params': {'step_size': 3, 'gamma': .5},
        'scheduler_params': {'mode': 'min', 'factor': 0.5, 'patience': 1},
        'optimizer_class': optim.Adam,
        'optimizer_params': {'lr': .0005},
        'loss_fn': PolicyLoss(calc_type=0, max_prob=1),
        'num_runs': 1,
        'train_fn': train_experiment
    },
    {
        'name': 'MNIST RL Policy (Adam, StepLR) RESNET18 calc_type=1',
        'model_fn': get_model_resnet,
        'model_params': {},
        'train_loader': train_loader,
        'val_loader': val_loader,
        'num_epochs': 20,
        'scheduler_class': ReduceLROnPlateau,
        # 'scheduler_params': {'step_size': 3, 'gamma': .5},
        'scheduler_params': {'mode': 'min', 'factor': 0.5, 'patience': 1},
        'optimizer_class': optim.Adam,
        'optimizer_params': {'lr': .0005},
        'loss_fn': PolicyLoss(calc_type=1, max_prob=1),
        'num_runs': 1,
        'train_fn': train_experiment
    },
    {
        'name': 'MNIST RL Policy (Adam, StepLR) RESNET18 calc_type=2',
        'model_fn': get_model_resnet,
        'model_params': {},
        'train_loader': train_loader,
        'val_loader': val_loader,
        'num_epochs': 20,
        'scheduler_class': ReduceLROnPlateau,
        # 'scheduler_params': {'step_size': 3, 'gamma': .5},
        'scheduler_params': {'mode': 'min', 'factor': 0.5, 'patience': 1},
        'optimizer_class': optim.Adam,
        'optimizer_params': {'lr': .0005},
        'loss_fn': PolicyLoss(calc_type=2, max_prob=1),
        'num_runs': 1,
        'train_fn': train_experiment
    },
    {
        'name': 'MNIST RL Policy (Adam, StepLR) RESNET18 calc_type=2',
        'model_fn': get_model_resnet,
        'model_params': {},
        'train_loader': train_loader,
        'val_loader': val_loader,
        'num_epochs': 20,
        'scheduler_class': ReduceLROnPlateau,
        # 'scheduler_params': {'step_size': 3, 'gamma': .5},
        'scheduler_params': {'mode': 'min', 'factor': 0.5, 'patience': 1},
        'optimizer_class': optim.Adam,
        'optimizer_params': {'lr': .0005},
        'loss_fn': PolicyLoss(calc_type=3, max_prob=1),
        'num_runs': 1,
        'train_fn': train_experiment
    },
    {
        'name': 'MNIST RL Policy (Adam, StepLR) RESNET18 calc_type=2',
        'model_fn': get_model_resnet,
        'model_params': {},
        'train_loader': train_loader,
        'val_loader': val_loader,
        'num_epochs': 20,
        'scheduler_class': ReduceLROnPlateau,
        # 'scheduler_params': {'step_size': 3, 'gamma': .5},
        'scheduler_params': {'mode': 'min', 'factor': 0.5, 'patience': 1},
        'optimizer_class': optim.Adam,
        'optimizer_params': {'lr': .0005},
        'loss_fn': PolicyLoss(calc_type=4, max_prob=1),
        'num_runs': 1,
        'train_fn': train_experiment
    },




    # {
    #     'name': 'MNIST RL Policy (Adam, StepLR) RESNET18 calc_type=1',
    #     'model_fn': get_model_resnet,
    #     'model_params': {},
    #     'train_loader': train_loader,
    #     'val_loader': val_loader,
    #     'num_epochs': 20,
    #     'scheduler_class': StepLR,
    #     'scheduler_params': {'step_size': 2, 'gamma': 0.75},
    #     'optimizer_class': optim.Adam,
    #     'optimizer_params': {'lr': .002},
    #     'loss_fn': PolicyLoss(calc_type=1, max_prob=1),
    #     'num_runs': 1,
    #     'train_fn': train_experiment
    # },




]


# Run experiments
for experiment in experiments:
    experiment_name = experiment['name']
    model_fn = experiment['model_fn']
    model_params = experiment['model_params']
    train_loader = experiment['train_loader']
    val_loader = experiment['val_loader']
    num_epochs = experiment['num_epochs']
    optimizer_class = experiment['optimizer_class']
    optimizer_params = experiment['optimizer_params']
    scheduler_class = experiment['scheduler_class']
    scheduler_params = experiment['scheduler_params']
    loss_fn = experiment['loss_fn']
    num_runs = experiment['num_runs']
    train_fn = experiment['train_fn']

    serialized_opt_params = " ".join(
        [f"{k} {str(v).replace('.', '_')}" for k, v in optimizer_params.items()])
    print(
        f"Running {experiment_name} {serialized_opt_params} for {num_runs} run(s)")

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        model = model_fn(**model_params)  # Create a new instance of the model
        model = model.to(device)

        train_fn(f"{experiment_name}", model, train_loader, val_loader, num_epochs,
                 optimizer_class, optimizer_params,  scheduler_class, scheduler_params, loss_fn, device, SummaryWriter)
    print(f"Finished {experiment_name} for {num_runs} run(s)")
