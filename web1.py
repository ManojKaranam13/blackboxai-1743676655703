import cv2
import os
import csv
import pandas as pd
import random
import torch
import re
import numpy as np
from PIL import Image
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torch import nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time
from typing import Dict
from math import nan
from math import isnan
import imageio.v2 as imageio

def seed_torch(seed=1111):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print ("Cuda device not found.")
    device = torch.device("mps")

print(device)

if torch.backends.mps.is_available(): device = torch.device("mps")
else: device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)


def parse_coordinate(value):
    value = value.strip('()')  # Remove the parentheses
    x_str, y_str = value.split(', ')
    return float(x_str), float(y_str)


#@title Normalization function
def normalize_coordinate(value, min_x, max_x, min_y, max_y):
    try:
        x, y = parse_coordinate(value)
        normalized_x = (x - min_x) / (max_x - min_x)
        normalized_y = (y - min_y) / (max_y - min_y)
        return normalized_x, normalized_y
    except ValueError as e:
        print(f"Skipping value: {value} due to error: {e}")
        return None, None

'''def minmax_normalization_data(df, min_x, max_x, min_y, max_y):
    for col in df.columns:
        if col.startswith('X'):
            # Extract X and Y values from each column and normalize them
            normalized_values = []
            for value in df[col]:
              if pd.isna(value):
                  normalized_values.append((0, 0))
              else:
                  normalized_x, normalized_y = normalize_coordinate(value, min_x, max_x, min_y, max_y)
                  normalized_values.append((normalized_x, normalized_y))
            df[col] = normalized_values
    return df'''
def minmax_normalization_data(df, min_x, max_x, min_y, max_y):
    for col in df.columns:
        if col.startswith('X'):
            normalized_values = []
            for value in df[col]:
                if pd.isna(value):
                    normalized_values.append((0, 0))
                else:
                    if isinstance(value, str):  # Check if value is a string
                        normalized_x, normalized_y = normalize_coordinate(value, min_x, max_x, min_y, max_y)
                    elif isinstance(value, tuple):  # Check if value is a tuple
                        normalized_x, normalized_y = normalize_coordinate(str(value), min_x, max_x, min_y, max_y)  # Convert tuple to string for parsing
                    else:
                        print(f"Skipping value: {value} in column: {col} due to unexpected type.")
                        normalized_x, normalized_y = (0, 0)  # or handle differently
                    normalized_values.append((normalized_x, normalized_y))
            df[col] = normalized_values
    return df



#@title Plotting function
def plot_trajectories(input_trajectory, target_trajectory, prediction_data, index):
    """
    Plot input + prediction and input + target trajectories.

    Args:
    - input_trajectory: Array of input trajectories, shape (num_samples, seq_length, 2)
    - target_trajectory: Array of target trajectories, shape (num_samples, seq_length, 2)
    - prediction_data: Dictionary of predicted trajectories, keys are IDs, values are trajectories
    - index: ID of the sample to plot
    """

    prediction_trajectory = prediction_data[index]
    input_trajectory, target_trajectory, prediction_trajectory = input_trajectory.to('cpu'), target_trajectory.to('cpu'), prediction_trajectory.to('cpu')

    input_trajectory = input_trajectory[(input_trajectory[:, 0] != 0) | (input_trajectory[:, 1] != 0)]
    target_trajectory = target_trajectory[(target_trajectory[:, 0] != 0) | (target_trajectory[:, 1] != 0)]
    prediction_trajectory = prediction_trajectory[(prediction_trajectory[:, 0] != 0) | (prediction_trajectory[:, 1] != 0)]

    plt.figure(figsize=(8, 6))

    # Plot input trajectory
    plt.plot(input_trajectory[:, 0], input_trajectory[:, 1], 'bo-', label='Input')
    # Plot prediction trajectory
    plt.plot(prediction_trajectory[:, 0], prediction_trajectory[:, 1], 'ro-', label='Prediction')
    # Plot target trajectory
    plt.plot(target_trajectory[:, 0], target_trajectory[:, 1], 'go-', label='Target')

    plt.title(f'Trajectory car {index}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid()

    # Set number formatting for x and y axes
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))

    plt.gca().grid(False)
    plt.show()
    
    


result = pd.read_csv(r'C:\Users\KMK\OneDrive\Documents\cbit\Capstone\final_project\dataset\filtered_test_trajectory_data_status1.csv')
x_values = []
y_values = []


# Loop through all columns in the dataframe
for col in result.columns:
    if col.startswith('X'):
        # Extract X and Y values from each column
        for value in result[col].dropna():  # Drop NaN values if any
            try:
                x, y = parse_coordinate(value)
                x_values.append(x)
                y_values.append(y)
            except ValueError as e:
                print(f"Skipping value: {value} in column: {col} due to error: {e}")

# Compute the minimum and maximum values
if x_values and y_values:
    test_min_x = min(x_values)
    test_max_x = max(x_values)
    test_min_y = min(y_values)
    test_max_y = max(y_values)

    print(f'Minimum X: {test_min_x}, Maximum X: {test_max_x}')
    print(f'Minimum Y: {test_min_y}, Maximum Y: {test_max_y}')

else:
    print("No valid X and Y values found.")
    
    
    
class VTPDataset(Dataset):
    def __init__(self, dataframe, image_folder, min_x, max_x, min_y, max_y,transform=None):
        self.dataframe = minmax_normalization_data(dataframe, min_x, max_x, min_y, max_y)
        self.image_folder = image_folder
        self.transform = transform
        self.device = torch.device(device)

        pattern_inputs1 = re.compile(r'X-\d+,Y-\d+')
        pattern_inputs2 = re.compile(r'X,Y')
        pattern_inputs3 = re.compile(r'X\+\d+,Y\+\d+')
        self.input_columns = [col for col in dataframe.columns if pattern_inputs1.match(col) or pattern_inputs2.match(col)]
        self.target_columns = [col for col in dataframe.columns if pattern_inputs3.match(col)]
        pattern_status = re.compile(r'^(STATUS|STATUS-\d+)$')
        self.status_columns = [col for col in dataframe.columns if pattern_status.match(col)]


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        frame = self.dataframe.iloc[idx]['Frames']
        img_name = os.path.join(self.image_folder, str(frame))
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        # Extract input coordinates (x, y)
        inputs_coordinates = self.dataframe.iloc[idx][self.input_columns]
        # Extract x and y coordinates separately
        x_values = inputs_coordinates.apply(lambda coord: coord[0]).values.astype(np.float32)
        y_values = inputs_coordinates.apply(lambda coord: coord[1]).values.astype(np.float32)
        # Create tensors for x and y coordinates
        x_values_array = np.array(x_values)
        y_values_array = np.array(y_values)
        inputs_coordinates = torch.tensor(np.array([x_values_array, y_values_array]), dtype=torch.float32).T

        # Extract input status (v, la, ta, a)
        inputs_status = self.dataframe.iloc[idx][self.status_columns].fillna('0.0,0.0,0.0,0.0')
        status_values = inputs_status.apply(lambda s: list(map(float, s.strip('()').split(',')))).values.tolist()
        inputs_status = torch.tensor(status_values, dtype=torch.float32)

        # Extract target coordinates (x, y)
        targets_coordinates = self.dataframe.iloc[idx][self.target_columns]
        x_values = targets_coordinates.apply(lambda coord: coord[0]).values.astype(np.float32)
        y_values = targets_coordinates.apply(lambda coord: coord[1]).values.astype(np.float32)
        # Create tensors for x and y coordinates
        target_coordinates = torch.tensor([x_values, y_values], dtype=torch.float32).T

        track_id = torch.tensor(self.dataframe.iloc[idx]['TRACK_ID'])

        return image.to(self.device), inputs_coordinates.to(self.device), inputs_status.to(self.device), target_coordinates.to(self.device), track_id.to(self.device)
    
    
    

class Trainer:
    """Utility class to train and evaluate a model."""
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler=None, log_steps: int = 1000, log_level: int = 2, epochs: int = 1):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.log_steps = log_steps
        self.log_level = log_level

    def loss_function(self, pred, target):
        """
        Args:
            pred: Tensor with predicted x, y coordinates (batch_size, seq_length, 2)
            target: Tensor with actual x, y coordinates (batch_size, seq_length, 2)
        Returns:
            ADE: float - Average Displacement Error
        """
        dist = torch.norm(pred - target, dim=-1)
        ade = dist.sum(dim=1)
        ade = ade.mean()
        return ade

    def train(self, train_dataloader: DataLoader, save_path: str = r'C:\Users\KMK\OneDrive\Documents\cbit\Capstone\final_project\Models\checkpointscheckpoint.pt', resume_from: str = None) -> dict[str, list[float]]:
        start_epoch = 1
        if resume_from:
            start_epoch, loss = self.load_checkpoint(resume_from)
            print(f"Resuming from checkpoint at epoch {start_epoch} with loss {loss}")

        if self.log_level > 0:
            print('Training ...')

        losses = {"train_losses": []}

        for epoch in range(start_epoch, self.epochs + 1):
            if self.log_level > 0:
                print(f'Epoch {epoch:2d}')

            epoch_start_time = time.time()
            epoch_loss = 0.0
            self.model.train()

            progress_bar = tqdm(train_dataloader, unit='batch')

            for step, (images, inputs_trajectory, inputs_status, target, id) in enumerate(progress_bar):
                self.optimizer.zero_grad()

                #print(inputs_trajectory.size())
                #print(inputs_status.size())
                #print(target.size())

                prediction = self.model(images, inputs_trajectory, inputs_status)
                sample_loss = self.loss_function(prediction, target)
                sample_loss.backward()
                self.optimizer.step()

                epoch_loss += sample_loss.item()

                if step % 100 == 0:
                    torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'loss': sample_loss}, f"{save_path}checkpoint.pt")

                if self.log_level > 1 and (step % self.log_steps) == (self.log_steps - 1):
                    progress_bar.set_description(f"Epoch {epoch}/{self.epochs}, Loss: {epoch_loss / (step + 1):.4f}")

            progress_bar.set_description(f"Epoch {epoch}/{self.epochs}, Loss: {epoch_loss / len(train_dataloader):.4f}")

            if self.scheduler:
                self.scheduler.step()

            epoch_end_time = time.time()
            print(f"\t[E: {epoch:2d}] Epoch time: {epoch_end_time - epoch_start_time:.2f} seconds")

            avg_epoch_loss = epoch_loss / len(train_dataloader)
            if self.log_level > 0:
                print(f'\t[E: {epoch:2d}] train loss = {avg_epoch_loss:.4f}')

            losses["train_losses"].append(avg_epoch_loss)

        if self.log_level > 0:
            print('... Done!')

        return losses

    def _compute_acc(self, pred, target) -> tuple[float, float]:
        ss_res = torch.sum((target - pred) ** 2)
        target_mean = torch.mean(target)
        ss_tot = torch.sum((target - target_mean) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2.item()

    def evaluate(self, valid_dataloader: DataLoader) -> tuple[float, float]:
        valid_loss = 0.0
        valid_acc = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch in valid_dataloader:
                images, inputs_trajectory, inputs_status, targets, id = batch
                predictions = self.model(images, inputs_trajectory, inputs_status)
                sample_loss = self.loss_function(predictions, targets)
                valid_loss += sample_loss.item()
                sample_acc = self._compute_acc(predictions, targets)
                valid_acc += sample_acc

        avg_valid_loss = valid_loss / len(valid_dataloader)
        avg_valid_acc = valid_acc / len(valid_dataloader)

        return avg_valid_loss, avg_valid_acc

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']

    def predict(self, dataloader: DataLoader) -> Dict:
        self.model.eval()
        predictions_dict = {}

        with torch.no_grad():
            for batch in dataloader:
                images, inputs_trajectory, inputs_status, targets, ids = batch
                predictions = self.model(images, inputs_trajectory, inputs_status)
                for prediction, id in zip(predictions, ids):
                    predictions_dict[id.item()] = prediction

        return predictions_dict



# prompt: any more changes to improve the performance of model

# prompt: upgrade the above code to improve accuracy

class VTPModel1(nn.Module):
    def __init__(self, input_size_trajectory, input_size_status, hidden_size, num_layers, device, target_lengths, cnn_path):
        super(VTPModel1, self).__init__()
        self.device = torch.device(device)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.target_lengths = target_lengths

        # Define CNN model
        self.cnn = models.resnet18(pretrained=True)  # Load pretrained weights
        self.cnn.fc = nn.Linear(512, 128)

        # Freeze CNN layers (optional, you can fine-tune if needed)
        for param in self.cnn.parameters():
            param.requires_grad = False


        # Define encoder and decoder
        self.encoder_traj = nn.LSTM(input_size=2, hidden_size=self.hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True) # Bidirectional LSTM for Trajectory
        self.encoder_status = nn.LSTM(input_size=4, hidden_size=self.hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True) # Bidirectional LSTM for Status
        self.decoder = nn.LSTM(input_size=(128+(self.hidden_size*2*2)), hidden_size=self.hidden_size * 2, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size * 2, 2)  # Output layer to produce (X, Y) tuples

    def forward(self, images, inputs_trajectory, inputs_status):

        batch_size = inputs_trajectory.size(0)

        # CNN branch
        cnn_output = self.cnn(images)  # [batch_size, 128]

        #Encoder
        encoder_traj_output, (hidden_traj, cell_traj) = self.encoder_traj(inputs_trajectory)
        encoder_status_output, (hidden_status, cell_status) = self.encoder_status(inputs_status)

        #Combine
        expanded_cnn_output = cnn_output.unsqueeze(1).expand(batch_size, encoder_traj_output.size(1), 128)
        combined_input = torch.cat((expanded_cnn_output, encoder_traj_output, encoder_status_output), dim=2)

        #Decoder
        decoder_hidden = torch.cat((hidden_traj[-2,:,:], hidden_status[-2,:,:]), dim=1).unsqueeze(0).repeat(self.num_layers, 1, 1) # Concatenate hidden states of last layer of both LSTMs (Bidirectional)
        decoder_cell = torch.cat((cell_traj[-2,:,:], cell_status[-2,:,:]), dim=1).unsqueeze(0).repeat(self.num_layers, 1, 1) # Concatenate cell states of last layer of both LSTMs (Bidirectional)

        decoder_output, (decoder_hidden, decoder_cell) = self.decoder(combined_input, (decoder_hidden, decoder_cell))

        # Output layer
        outputs = torch.zeros((batch_size, self.target_lengths, 2), device=self.device)

        for t in range(self.target_lengths):
            feed = decoder_output[:, t, :]
            _decoder_output = self.fc(feed)
            outputs[:, t, :] = _decoder_output

        return outputs

# Define paths to image folder and CSV files
image_folder = r'C:\Users\KMK\OneDrive\Documents\cbit\Capstone\final_project\frames'
test_image_folder = r'C:\Users\KMK\OneDrive\Documents\cbit\Capstone\final_project\testing_videos'
train_file = r'C:\Users\KMK\OneDrive\Documents\cbit\Capstone\final_project\dataset\filtered_trajectorydata_status.csv'
test_file =filepath

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((160, 90)),                                               # Resize images to a fixed size
    transforms.ToTensor(),                                                      # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize images
])

# Load dataset from CSV
df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)

# Create custom dataset
test_dataset = VTPDataset(df_test, test_image_folder, test_min_x, test_max_x, test_min_y, test_max_y, transform=transform)

# Create DataLoaders
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)



#Load the model
input_size_trajectory = 2
input_size_status = 4
target_lengths = 5
hidden_size = 128
num_layers = 2
cnn_path = r"C:\Users\KMK\OneDrive\Documents\cbit\Capstone\final_project\Models\resnet18-f37072fd.pth"
save_path_vtp = r'C:\Users\KMK\OneDrive\Documents\cbit\Capstone\final_project\Models\VTPModel2.pth'

loaded_model = VTPModel1(input_size_trajectory, input_size_status, hidden_size, num_layers, device, target_lengths, cnn_path)
loaded_model.load_state_dict(torch.load(save_path_vtp, map_location=torch.device(device)))
loaded_model.to(device)
    
    
    
#optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
optimizer = optim.Adam(loaded_model.parameters(), lr=0.0005, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10)
num_epochs = 50

#trainer1 = Trainer(model=model, optimizer=optimizer,scheduler=scheduler, epochs=num_epochs)
trainer1 = Trainer(model=loaded_model, optimizer=optimizer,scheduler=scheduler, epochs=num_epochs)



#trainer1 = Trainer(model, optimizer, num_epochs)
trainer1 = Trainer(loaded_model, optimizer, num_epochs)
predictions = trainer1.predict(test_dataloader)

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Miss Rate & Accuracy Threshold (in meters)
threshold = 0.25  

# Initialize metrics
total_ade, total_fde, total_rmse, total_mae, total_r2 = 0, 0, 0, 0, 0
miss_count = 0
correct_predictions = 0  # For accuracy calculation
num_samples = 0
hde_metrics = {t: 0.0 for t in [1, 3, 5]}  # Horizon-wise ADE

# Loop through the test dataset
for i in range(len(test_dataset)):
    # Get test sample
    image_test, input_trajectory, status_test, target_trajectory, id_test = test_dataset.__getitem__(i)
    id_test = id_test.item()  # Ensure ID is integer

    # Move to CPU
    input_trajectory, target_trajectory = input_trajectory.to('cpu'), target_trajectory.to('cpu')
    prediction_trajectory = predictions[id_test].to('cpu')  # Get corresponding prediction

    # Remove zero-coordinate values
    input_trajectory = input_trajectory[(input_trajectory[:, 0] != 0) | (input_trajectory[:, 1] != 0)]
    target_trajectory = target_trajectory[(target_trajectory[:, 0] != 0) | (target_trajectory[:, 1] != 0)]
    prediction_trajectory = prediction_trajectory[(prediction_trajectory[:, 0] != 0) | (prediction_trajectory[:, 1] != 0)]

    # Ensure same sequence length
    min_len = min(len(target_trajectory), len(prediction_trajectory))
    target_trajectory = target_trajectory[:min_len]
    prediction_trajectory = prediction_trajectory[:min_len]

    # Convert to NumPy
    target_np = target_trajectory.numpy()
    pred_np = prediction_trajectory.numpy()

    # Compute ADE (Average Displacement Error)
    ade = torch.norm(prediction_trajectory - target_trajectory, dim=1).mean().item()
    total_ade += ade

    # Compute FDE (Final Point Distance)
    fde = torch.norm(prediction_trajectory[-1] - target_trajectory[-1]).item()
    total_fde += fde

    # Compute RMSE and MAE
    rmse = np.sqrt(mean_squared_error(target_np, pred_np))
    mae = mean_absolute_error(target_np, pred_np)
    total_rmse += rmse
    total_mae += mae

    # Compute R² Score
    r2 = r2_score(target_np, pred_np)
    total_r2 += r2

    # Compute Miss Rate
    if fde > threshold:
        miss_count += 1
    else:
        correct_predictions += 1  # Count correct predictions

    # Compute Horizon-wise Displacement Error (HDE)
    for t in hde_metrics.keys():
        if t < len(target_trajectory):
            hde_metrics[t] += torch.norm(prediction_trajectory[t] - target_trajectory[t]).item()

    num_samples += 1

    # ---- PLOT TRAJECTORIES ----
    plot_trajectories(input_trajectory, target_trajectory, predictions, id_test)

    # ---- PRINT METRICS FOR EACH TRAJECTORY ----
    print(f"\nTrajectory {id_test}:")
    print(f"ADE: {ade:.4f} meters, FDE: {fde:.4f} meters")
    print(f"RMSE: {rmse:.4f} meters, MAE: {mae:.4f} meters, R²: {r2:.4f}")
    print(f"Miss Rate (> {threshold}m): {'YES' if fde > threshold else 'NO'}\n")

# ---- FINAL AVERAGED METRICS ----
ade_final = total_ade / num_samples
fde_final = total_fde / num_samples
rmse_final = total_rmse / num_samples
mae_final = total_mae / num_samples
r2_final = total_r2 / num_samples
miss_rate = miss_count / num_samples * 100  # Convert to percentage
accuracy = correct_predictions / num_samples * 100  # Convert to percentage
hde_final = {t: hde_metrics[t] / num_samples for t in hde_metrics.keys()}

# ---- PRINT FINAL RESULTS ----
print("\n===== FINAL EVALUATION RESULTS =====")
print(f"Average Displacement Error (ADE): {ade_final:.4f} meters")
print(f"Final Displacement Error (FDE): {fde_final:.4f} meters")
print(f"Root Mean Squared Error (RMSE): {rmse_final:.4f} meters")
print(f"Mean Absolute Error (MAE): {mae_final:.4f} meters")
print(f"R² Score: {r2_final:.4f}")
print(f"Miss Rate (> {threshold}m): {miss_rate:.2f}%")
import sys
sys.stdout.reconfigure(encoding='utf-8')
print(f"Prediction Accuracy (FDE ≤ {threshold}m): {accuracy:.2f}%")

for t, hde_value in hde_final.items():
    print(f"Horizon-wise Displacement Error (HDE) at {t}s: {hde_value:.4f} meters")
