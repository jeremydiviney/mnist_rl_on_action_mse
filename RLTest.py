import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

print("PyTorch version:", torch.__version__)

print(torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("GPU details:", torch.cuda.get_device_properties(0))  # Details for GPU index 0

print(torch.cuda.get_device_name(0))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
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


model = PolicyNetwork().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.003)

scheduler = StepLR(optimizer, step_size=5, gamma=0.66)


# Load MNIST data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=256, shuffle=True)
val_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, download=True, transform=transform), batch_size=256, shuffle=False)

# Training loop
for epoch in range(1, 40):  # 10 epochs
    total_loss = 0
    total_correct = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)

        # Sample action
        # action = torch.multinomial(outputs, 1).squeeze()
        _, action = outputs.max(dim=1)

        # Calculate reward
        # reward = ((action == targets).float() * 2 - 1)

        # Making the loss a differentiable computation
        # loss = (action_probabilities.log()[range(action_probabilities.shape[0]), action] * reward).mean()

        # loss = (outputs.log()[range(outputs.shape[0]), action] * reward).mean()

        # loss = (outputs[range(outputs.shape[0]), action] - reward).mean()

        # Calculate the reward
        reward = (action == targets).float() * 2 - 1  # 1 for correct, -1 for incorrect

        # MSE loss
        # chosen_action_values = outputs[torch.arange(outputs.size(0)), action]
        # loss = F.mse_loss(chosen_action_values, chosen_action_values + reward)

        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(outputs, targets)

        # Applying the reward
        # loss = (-reward * loss).mean()

        total_loss += loss.item()

        # loss = F.loss(chosen_action_values, chosen_action_values + reward)

        total_correct += (action == targets).float().sum().item()

        loss.backward()
        optimizer.step()

    scheduler.step()
    print('Epoch: {}, Loss: {:.3f}'.format(epoch, total_loss / len(train_loader)),
          'Accuracy: {:.3f}'.format(total_correct / len(train_loader.dataset)))
