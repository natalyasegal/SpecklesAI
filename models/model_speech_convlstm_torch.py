import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Normalize
import numpy as np
import random

def set_seed(seed=9):
    """Set all seeds to ensure reproducibility."""
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU.

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ConvLSTMModel(nn.Module):
    def __init__(self, config, sz_conv, sz_dense):
        super(ConvLSTMModel, self).__init__()
        self.norm1 = nn.BatchNorm2d(1)  # Assuming input channel is 1
        self.convlstm = nn.Conv2d(1, sz_conv, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(sz_conv)
        self.dropout1 = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(sz_conv * config.frame_size_x * config.frame_size_y, sz_dense)
        self.norm3 = nn.BatchNorm1d(sz_dense)
        self.dropout2 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(sz_dense, config.number_of_classes)
        self.activation = nn.Softmax(dim=1) if config.number_of_classes > 2 else nn.Sigmoid()

    def forward(self, x):
        x = self.norm1(x)
        x = torch.relu(self.convlstm(x))
        x = self.norm2(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = torch.relu(self.dense1(x))
        x = self.norm3(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        x = self.activation(x)
        return x

def train_model(config, sz_conv, sz_dense, x_train, y_train, x_val, y_val, batch_sz, n_epochs):
    # Create Dataset and DataLoader
    train_dataset = TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).long())
    train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)
    val_dataset = TensorDataset(torch.tensor(x_val).float(), torch.tensor(y_val).long())
    val_loader = DataLoader(val_dataset, batch_size=batch_sz, shuffle=False)

    model = ConvLSTMModel(config.number_of_classes, config.frame_size_x, config.frame_size_y, sz_conv, sz_dense)
    criterion = nn.CrossEntropyLoss() if config.number_of_classes > 2 else nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    # Training loop
    best_accuracy = 0
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, torch.max(labels, 1)[1] if config.number_of_classes > 2 else labels.view(-1, 1).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation accuracy
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == torch.max(labels, 1)[1]).sum().item()

        val_accuracy = correct / total
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')  # Save the best model

        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}, Validation Accuracy: {val_accuracy}')

    model.load_state_dict(torch.load('best_model.pth'))  # Load the best model
    return model

def load_model(config): 
    return tf.keras.models.load_model(config.models_path)
    
