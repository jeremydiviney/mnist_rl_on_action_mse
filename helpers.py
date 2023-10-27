import math
import random
import time
import torch
import torch.nn as nn


def log_layer_histogram(engine, model, epoch, writer):
    for name, param in model.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)


def log_grad_histogram(engine, model, epoch, writer):
    for name, param in model.named_parameters():
        writer.add_histogram(
            name + '-grad', param.grad.clone().cpu().data.numpy(), epoch)


def train_experiment(experiment_name, model, train_loader, val_loader, num_epochs, optimizer_class, optimizer_params, scheduler_class, scheduler_params, loss_fn,  device, SummaryWriter):
    # Serialize optimizer parameters
    serialized_opt_params = " ".join(
        [f"{k} {str(v).replace('.', '_')}" for k, v in optimizer_params.items()])
    writer = SummaryWriter(
        log_dir=f'logs/{experiment_name}_({serialized_opt_params})')

    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    scheduler = scheduler_class(optimizer, **scheduler_params) if scheduler_class else None

    for epoch in range(num_epochs):

        # Training
        model.train()
        total_loss = 0.0
        correct_train = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)

            _, pred = outputs.max(1)
            correct_pred = targets
            correct_train += pred.eq(correct_pred).sum().item()

        avg_train_loss = total_loss / len(train_loader.dataset)
        train_accuracy = correct_train / len(train_loader.dataset)

        writer.add_scalar('train/loss', avg_train_loss, epoch)
        writer.add_scalar('train/accuracy', train_accuracy, epoch)
        print(
            f"T - Epoch: {epoch}, Loss: {avg_train_loss:.7f}, Accuracy: {train_accuracy:.4f}")

        # Validation
        model.eval()
        total_loss = 0.0
        correct_val = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                total_loss += loss.item() * inputs.size(0)

                _, pred = outputs.max(1)
                correct_pred = targets
                correct_val += pred.eq(correct_pred).sum().item()

        avg_val_loss = total_loss / len(val_loader.dataset)
        val_accuracy = correct_val / len(val_loader.dataset)

        writer.add_scalar('val/loss', avg_val_loss, epoch)
        writer.add_scalar('val/accuracy', val_accuracy, epoch)
        print(
            f"V - Epoch: {epoch}, Loss: {avg_val_loss:.7f}, Accuracy: {val_accuracy:.4f}")

        if scheduler:
            scheduler.step()

    print(list(dict(model.named_parameters()).keys()))

    writer.close()
