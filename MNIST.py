import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

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


def propability_onehot_transform(output):
    y_pred, target = output
    predicted_indices = torch.argmax(y_pred, dim=1)
    y_pred_onehot = torch.eye(y_pred.size(1), device=y_pred.device)[
        predicted_indices]
    return y_pred_onehot, target


class DenseLinearModel(nn.Module):
    def __init__(self):
        super(DenseLinearModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.softmax(self.fc5(x), dim=1)
        return x


def get_model():
    return DenseLinearModel()


num_classes = 10


train_loader, test_loader = get_loaders(num_classes=num_classes, batch_size=512)


# Define your experiments
experiments = [
    {
        'name': 'MNIST Binary Gradient Step Walk (adaptive order of magnitude buckets)',
        'model': get_model,
        'model_params': {'num_classes': num_classes},
        'train_loader': train_loader,
        'test_loader': test_loader,
        'num_epochs': 100,
        'scheduler_class': StepLR,
        'scheduler_params': {'step_size': 5, 'gamma': 0.5},
        'optimizer_class': optim.Adam,
        'optimizer_params': {'lr': .005},
        'loss_FN': nn.CrossEntropyLoss(),
        'num_runs': 1,
        'train_fn': train_experiment
    },

]


# Run experiments
for experiment in experiments:
    experiment_name = experiment['name']
    model = experiment['model']
    model_params = experiment['model_params']
    train_loader = experiment['train_loader']
    test_loader = experiment['test_loader']
    num_epochs = experiment['num_epochs']
    optimizer_class = experiment['optimizer_class']
    optimizer_params = experiment['optimizer_params']
    scheduler_class = experiment['scheduler_class']
    scheduler_params = experiment['scheduler_params']
    loss_FN = experiment['criterion']
    accuracy_metric = experiment['accuracy_metric']
    num_runs = experiment.get('num_runs', 1)
    train_fn = experiment.get('train_fn', train_experiment)
    serialized_opt_params = " ".join(
        [f"{k} {str(v).replace('.', '_')}" for k, v in optimizer_params.items()])
    print(
        f"Running {experiment_name} {serialized_opt_params} for {num_runs} run(s)")

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        model = get_model(**model_params)  # Create a new instance of the model
        model = model.to(device)

        train_fn(f"{experiment_name}", model, train_loader, test_loader, num_epochs,
                 optimizer_class, optimizer_params, loss_FN, scheduler_class, device, SummaryWriter)
    print(f"Finished {experiment_name} for {num_runs} run(s)")
