import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import savgol_filter, butter, filtfilt
def add_noise(x, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x)
    alpha = 1 - 0.01 * t.view(-1, 1, 1) / 1000
    return alpha * x + (1 - alpha) * noise, noise

def Denoise(model, shape, steps=100):
    model.eval()
    x = segments[0]
    x[0,0][130:160] = -0.04
    x = x.to(device)  # bruit initial
    for t in reversed(range(steps)):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.float32)
        noise_pred = model(x, t_batch)
        alpha = 1 - 0.01 * t / 1000
        x = (x - (1 - alpha) * noise_pred) / alpha
       
    return x




def savgol_denoise(signal, window=11, poly=2):
    return savgol_filter(signal, window_length=window, polyorder=poly)

def butter_lowpass_filter(data, cutoff=3, fs=50, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def train_model(model, dataloader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        for noisy, clean in dataloader:
            noisy, clean = noisy.to(device), clean.to(device)
            noisy = noisy.permute(0, 2, 1)
            clean = clean.permute(0, 2, 1)
            output = model(noisy)
            loss = loss_fn(output, clean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")