
# Author: Neel Kanwal, neel.kanwal0@gmail.com
# This is AI-powered forecasting, uses previouly trained transformer model weights to extrapolate new data based on the information. 


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os
import matplotlib
import matplotlib.pyplot as plt

font = {'family' : 'serif',
        'weight':'normal',
        'size'   : 28}
matplotlib.rc('font', **font)
plt.rcParams["figure.figsize"] = (24, 16)
matplotlib.use('Agg')

import seaborn as sns
sns.set_style("white")
import pandas as pd
import numpy as np
from nixtla import NixtlaClient
from torch.utils.data import DataLoader
import torch

from myfunctions import FHRDataset_v2, TransformerMaskedAutoencoder_inference, FHRDataset_v4_envelope

###
torch.cuda.empty_cache()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
cuda_device = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
torch.cuda.empty_cache()
# torch.cuda.set_device(cuda_device)
device = torch.cuda.current_device()
print("Current CUDA device = ", device)
###

# Perform recursive forecasting
start_index = 3600
end_index = 7200
step_size = 30
window_size = 3600
##

val_dir = "/.../validation_interpolated_with_envelope/"


val_dataset = FHRDataset_v4_envelope(val_dir, sequence_length=7200)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

def get_sample_at_index(dataloader, index):
    dataset = dataloader.dataset
    sample = dataset[index]
    # print(f"Retrieving sample at index {index}")
    # print(f"First few values of the sample: {sample[0][:5]}")  # Assuming the first element is the time series
    return sample

def sample_to_dataframe(sample):
    time_series, outcomes = sample
    hybrid_30_signal = time_series[1] # Using 5 for hybrid-30. see FHRDataset_v4_envelope loader code
    hybrid_30_signal = hybrid_30_signal.squeeze().numpy()
    df = pd.DataFrame(hybrid_30_signal, columns=['value'])
    df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='S')
    return df, outcomes


def recursive_forecast_timegpt(df, start_index, end_index, step_size, window_size=window_size):
    forecasts = []
    extended_df = df.copy()

    for i in range(start_index, end_index, step_size):
        # Use a sliding window of recent history
        window_size = min(window_size, i)  # Use at most 1200 recent points
        train_df = extended_df.iloc[i-window_size:i]
        
        try:
            # Generate forecast
            forecast_df = nixtla_client.forecast(train_df, h=step_size, time_col='timestamp', target_col='value', model='timegpt-1-long-horizon')
            
            # Extract the forecasted values
            new_forecasts = forecast_df['TimeGPT'].values
            forecasts.extend(new_forecasts)
            
            # Add the new forecasts to the extended dataframe
            new_data = pd.DataFrame({'timestamp': range(i, i+step_size), 'value': new_forecasts})
            extended_df = pd.concat([extended_df, new_data], ignore_index=True)

        except Exception as e:
            print(f"Error occurred at index {i}: {str(e)}")
            break

    return pd.DataFrame({'forecast': forecasts})

def recursive_forecast_fhrtransformer(df, start_index, end_index, step_size, model, device, window_size=window_size):
    model = model.to(device)
    model.eval()
    forecasts = []
    extended_df = df.copy()

    for i in range(start_index, end_index, step_size):
        # Use a sliding window of recent history
        window_size = min(window_size, i)  # Use at most window_size recent points
        input_data = extended_df['value'].values[i-window_size:i]
        
        # Ensure input_data is divisible by patch_size
        if len(input_data) % model.patch_size != 0:
            pad_length = model.patch_size - (len(input_data) % model.patch_size)
            input_data = np.pad(input_data, (pad_length, 0), 'edge')
        
        # Reshape input_data to (batch_size, seq_len, input_dim)
        input_tensor = torch.FloatTensor(input_data).unsqueeze(0).unsqueeze(-1).to(device)
        
        try:
            with torch.no_grad():
                # Predict next step_size values
                output = model(input_tensor, input_tensor)
                forecast = output[0, -step_size:, 0].cpu().numpy()
                
                forecasts.extend(forecast)
                
                # Add the new forecasts to the extended dataframe
                new_data = pd.DataFrame({'timestamp': range(i, i+step_size), 'value': forecast})
                extended_df = pd.concat([extended_df, new_data], ignore_index=True)
        except Exception as e:
            print(f"Error occurred at index {i}: {str(e)}")
            break
    
    return pd.DataFrame({'forecast': forecasts})
    

# TimeGPT
# key = 'nixa------------215FR7d9IroVACU7xh5LqXo'
# nixtla_client = NixtlaClient(api_key = key)
# nixtla_client.validate_api_key()

# Transformer with simple loss
model_sim = TransformerMaskedAutoencoder_inference(input_dim=1, d_model=512, nhead=16, num_encoder_layers=5,num_decoder_layers=5, dim_feedforward=1024, patch_size=30)
# best_model_wts = '/......../10_25_2024_12_47_52/best_weights.dat'
best_model_wts = '/......./12_01_2024_02_55_50/best_weights.dat'
model_sim.load_state_dict(torch.load(best_model_wts, map_location=torch.device('cpu'))['model'])


# Transformer with hybrid loss
model_hybrid = TransformerMaskedAutoencoder_inference(input_dim=1, d_model=512, nhead=16, num_encoder_layers=5,num_decoder_layers=5, dim_feedforward=1024, patch_size=60)
best_model_wts = '/.../10_26_2024_04_25_07/best_weights.dat'
model_hybrid.load_state_dict(torch.load(best_model_wts, map_location=torch.device('cpu'))['model'])

# model_hybrid = TransformerMaskedAutoencoder_inference(input_dim=1, d_model=512, nhead=16, num_encoder_layers=5,num_decoder_layers=5, dim_feedforward=1024, patch_size=30)
# best_model_wts = '............/12_01_2024_02_00_36/best_weights.dat'
# model_hybrid.load_state_dict(torch.load(best_model_wts, map_location=torch.device('cpu'))['model'])


#153
for index in [0, 6, 105, 153, 108,163, 177]:
    # Extract a sample to perform forecasting
    sample = get_sample_at_index(val_loader, index=index)
    df, outcomes = sample_to_dataframe(sample)
    # outcome_30min, outcome_24hours, outcome_resus = outcomes.tolist()

    # ## TIMEGPT Forecast here
    # print("### Forecasting using TimeGPT ###")
    # forecast_results = recursive_forecast_timegpt(df, start_index, end_index, step_size)

    ## FHR Transformer Forecast here, autoregressive mode
    print("### Forecasting using FHR Transformer (Hybrid-30) ###")
    forecast_results_fhrtransformer = recursive_forecast_fhrtransformer(df, start_index, end_index, step_size, model_sim, device)

    ## FHR Transformer Forecast here, autoregressive mode
    print("### Forecasting using FHR Transformer (Hybrid-60) ###")
    forecast_results_fhrtransformer_hybrid = recursive_forecast_fhrtransformer(df, start_index, end_index, step_size, model_hybrid, device)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 10), sharex=True)
    fig.suptitle(f"True Values and Forecast Approaching the Event\n", fontsize=24)

    data_length = len(df['value'])
    x_axis = np.arange(-data_length+1, 1)
    forecast_length = len(forecast_results_fhrtransformer)
    forecast_x = np.arange(-forecast_length+1, 1)

    # Function to plot on each subplot
    def plot_subplot(ax, true_values, forecast, label, color):
        ax.plot(x_axis, true_values, label='True Values', color='blue', linewidth=2)
        ax.plot(forecast_x, forecast, label=label, color=color, linewidth=1, alpha=0.7)
        rolling_variance = forecast.rolling(window=step_size).var()
        confidence_interval = 1.96 * np.sqrt(rolling_variance)
        upper_bound = forecast + confidence_interval
        lower_bound = forecast - confidence_interval
        ax.fill_between(forecast_x, lower_bound, upper_bound, color=color, alpha=0.2, label='95% Confidence Interval')

        # Vertical line at -3600
        ax.axvline(x=-3600, color='black', linestyle='--', linewidth=2, label='Forecast Starts')
    
        # Shade region between -7200 and -3600
        ax.axvspan(-3600-window_size, -3600, alpha=0.15, color='gray', label='Input Region')
        ax.set_ylim(-0.2, 1.2)
        ax.set_ylabel('Value', fontsize=18)
        # ax.legend(fontsize=14, loc='lower left')
        ax.grid(True, linestyle='--', alpha=0.6)


    # Plot FetalFormer (Simple)
    plot_subplot(ax1, df['value'], forecast_results_fhrtransformer['forecast'], 'FetalTransformer Hybrid-30-Sch', 'green')
    ax1.set_title('FetalTransformer Forecast (Simple)', fontsize=20)

    # Plot FetalFormer (Hybrid)
    plot_subplot(ax2, df['value'], forecast_results_fhrtransformer_hybrid['forecast'], 'FetalTransformer Hybrid-Old', 'orange')
    ax2.set_title('FetalTransformer Forecast (Hybrid)', fontsize=20)

    # Set common x-label
    fig.text(0.5, 0.01, 'Time to event (seconds)', ha='center', fontsize=18)

    plt.tight_layout()
    plt.savefig(f"5_Forecast_comparison_{index}.png", dpi=300, bbox_inches='tight')
    plt.show()

