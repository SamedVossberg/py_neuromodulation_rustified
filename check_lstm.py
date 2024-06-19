import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics

# Custom Dataset
class CustomSequenceDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sequence = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sequence = self.transform(sequence)

        label = torch.tensor(label, dtype=torch.float32)

        return sequence, label

class ToTensor(object):
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=4):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

# Training function
def train_model(model, dataloader, num_epochs=20, learning_rate=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(sequences)
            outputs = outputs.view(-1)
            loss = criterion(outputs, labels.float())
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * sequences.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

def predict(model, dataloader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for sequences, _ in dataloader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            outputs = outputs.view(-1)
            predictions.extend(outputs.cpu().numpy())

    return np.array(predictions)

if __name__ == "__main__":
    # Sample data: n_sequences x seq_length x input_size
    data = np.random.rand(1000, 30, 10)  # Replace with your actual data
    labels = np.random.randint(0, 2, size=(1000,))  # Replace with your actual binary labels

    data_raw = np.load(r"E:\Downloads\all_merged_normed_rmap.npy", allow_pickle=True)
    labels = data_raw[:, -1, -3].astype(float)
    data = data_raw[:, :, :-4].astype(float)

    for sub_idx in np.unique(data_raw[:, -1, -1]):
        train_idx = np.where(data_raw[:, -1, -1] != sub_idx)
        test_idx = np.where(data_raw[:, -1, -1] == sub_idx)

        X_train = data[train_idx]
        y_train = labels[train_idx]
        X_test = data[test_idx]
        y_test = labels[test_idx]

        # Define the transformation
        transform = ToTensor()

        # Create the dataset
        dataset_train = CustomSequenceDataset(data=X_train, labels=y_train, transform=transform)
        dataset_test = CustomSequenceDataset(data=X_test, labels=y_test, transform=transform)

        # Create the DataLoader
        dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=1)
        data_test = DataLoader(dataset_test, batch_size=32, shuffle=False, num_workers=1)

        # Define model parameters
        input_size = data.shape[2]  # This should match the third dimension of your data
        hidden_size = 50
        num_layers = 1

        # Create the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMModel(input_size, hidden_size, num_layers).to(device)

        # Train the model
        model = train_model(model, dataloader_train, num_epochs=20, learning_rate=0.001)
        
        # predict the test data
        test_predictions = predict(model, data_test)

        print(f"ba: {metrics.balanced_accuracy_score(y_test, test_predictions)}")