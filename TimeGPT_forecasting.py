# Author: Neel Kanwal, neel.kanwal0@gmail.
# Obtain TimeGPT API token to use it.
# Forecasting for comparing. 

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
cuda_device = 6
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


# val_dataset = FHRDataset_v2(val_dir, sequence_length=7200, normalization='minmax', force_norm=True)
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
# # outcome_dic = {0: "Normal", 1:"NICU", 3:"Death"}

# def get_sample_at_index(dataloader, index):
#     dataset = dataloader.dataset
#     sample = dataset[index]
#     # print(f"Retrieving sample at index {index}")
#     # print(f"First few values of the sample: {sample[0][:5]}")  # Assuming the first element is the time series
#     return sample

# def sample_to_dataframe(sample):
#     time_series, outcomes = sample
#     time_series = time_series.squeeze().numpy()
#     df = pd.DataFrame(time_series, columns=['value'])
#     df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='S')
#     return df, outcomes

val_dir = "/nfs/....../validation_interpolated_with_envelope/"


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
    hybrid_30_signal = time_series[1] # Using 5 for hybrid-30. see FHRDataset_v4_envelope loader code,  Using 1 for linearly interpolated version
    hybrid_30_signal = hybrid_30_signal.squeeze().numpy()
    df = pd.DataFrame(hybrid_30_signal, columns=['value'])
    df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='S')
    return df, outcomes

# def recursive_forecast_timegpt(df, start_index, end_index, step_size):
#     forecasts = []
#     extended_df = df.copy()

#     for i in range(start_index, end_index, step_size):
#         # Use all available data (original + forecasted) up to this point
#         train_df = extended_df.iloc[:i]
        
#         # Generate forecast
#         forecast_df = nixtla_client.forecast(train_df, h=step_size, time_col='timestamp', target_col='value', model='timegpt-1-long-horizon')
        
#         # Extract the forecasted values
#         new_forecasts = forecast_df['TimeGPT'].values
#         forecasts.extend(new_forecasts)
        
#         # Add the new forecasts to the extended dataframe
#         new_data = pd.DataFrame({'timestamp': range(i, i+step_size), 'value': new_forecasts})
#         extended_df = pd.concat([extended_df, new_data], ignore_index=True)

#     return pd.DataFrame({'forecast': forecasts})

# def recursive_forecast_fhrtransformer(df, start_index, end_index, step_size, model, device):
#     model = model.to(device)
#     model.eval()
#     forecasts = []
#     time_series = df['value'].values[:start_index].tolist()  # Convert to list for easy appending
    
#     for i in range(start_index, end_index, step_size):
#         # Prepare input data (including previously forecasted values)
#         input_data = np.array(time_series)
        
#         # Pad input_data if necessary
#         if len(input_data) % model.patch_size != 0:
#             pad_length = model.patch_size - (len(input_data) % model.patch_size)
#             input_data = np.pad(input_data, (0, pad_length), 'constant', constant_values=0)
        
#         # Reshape input_data to (batch_size, seq_len, input_dim)
#         input_tensor = torch.FloatTensor(input_data).unsqueeze(0).unsqueeze(-1).to(device)
        
#         with torch.no_grad():
#             # Predict next step_size values
#             output = model(input_tensor, input_tensor)  # No mask needed
#             forecast = output[0, -step_size:, 0].cpu().numpy()
#             forecasts.extend(forecast)
            
#             # Append forecasted values to time_series for next iteration
#             time_series.extend(forecast)
    
#     return pd.DataFrame({'forecast': forecasts})

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
key = '...jP...................................................oD9zOs6X'
nixtla_client = NixtlaClient(api_key = key)
nixtla_client.validate_api_key()

# Transformer with simple loss
model_sim = TransformerMaskedAutoencoder_inference(input_dim=1, d_model=512, nhead=16, num_encoder_layers=5,num_decoder_layers=5, dim_feedforward=1024, patch_size=30)
best_model_wts = '/nfs/.................../10_25_2024_12_47_52/best_weights.dat'
model_sim.load_state_dict(torch.load(best_model_wts, map_location=torch.device('cpu'))['model'])


# Transformer with hybrid loss
model_hybrid = TransformerMaskedAutoencoder_inference(input_dim=1, d_model=512, nhead=16, num_encoder_layers=5,num_decoder_layers=5, dim_feedforward=1024, patch_size=60)
best_model_wts = '/nfs/......../10_26_2024_04_25_07/best_weights.dat'
model_hybrid.load_state_dict(torch.load(best_model_wts, map_location=torch.device('cpu'))['model'])

#153
for index in [0,104,153]:
    # Extract a sample to perform forecasting
    sample = get_sample_at_index(val_loader, index=index)
    df, outcomes = sample_to_dataframe(sample)
    outcome_30min, outcome_24hours, outcome_resus = outcomes.tolist()

    ## TIMEGPT Forecast here
    print("### Forecasting using TimeGPT ###")
    forecast_results = recursive_forecast_timegpt(df, start_index, end_index, step_size)

    ## FHR Transformer Forecast here, autoregressive mode
    print("### Forecasting using FHR Transformer (trained with simple loss) ###")
    forecast_results_fhrtransformer = recursive_forecast_fhrtransformer(df, start_index, end_index, step_size, model_sim, device)

    ## FHR Transformer Forecast here, autoregressive mode
    print("### Forecasting using FHR Transformer (trained with hybrid loss) ###")
    forecast_results_fhrtransformer_hybrid = recursive_forecast_fhrtransformer(df, start_index, end_index, step_size, model_hybrid, device)

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(25, 15), sharex=True)
    fig.suptitle(f"True Values and Forecast Approaching the Event\n",  fontsize=24)
                 # f"Outcomes: 30min={outcome_dic[outcome_30min]}, 24h={outcome_dic[outcome_24hours]}, Resus={outcome_resus}", 
                

    data_length = len(df['value'])
    x_axis = np.arange(-data_length+1, 1)
    forecast_length = len(forecast_results['forecast'])
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
        ax.legend(fontsize=14, loc='lower left')
        ax.grid(True, linestyle='--', alpha=0.6)

    # Plot TimeGPT
    plot_subplot(ax1, df['value'], forecast_results['forecast'], 'TimeGPT Forecast', 'red')
    ax1.set_title('TimeGPT Forecast', fontsize=20)

    # Plot FetalFormer (Simple)
    plot_subplot(ax2, df['value'], forecast_results_fhrtransformer['forecast'], 'FetalTransformer Hybrid-30', 'green')
    ax2.set_title('FetalTransformer Forecast (Simple)', fontsize=20)

    # Plot FetalFormer (Hybrid)
    plot_subplot(ax3, df['value'], forecast_results_fhrtransformer_hybrid['forecast'], 'FetalTransformer Hybrid-60', 'orange')
    ax3.set_title('FetalTransformer Forecast (Hybrid)', fontsize=20)

    # Set common x-label
    fig.text(0.5, 0.01, 'Time to event (seconds)', ha='center', fontsize=18)

    plt.tight_layout()
    plt.savefig(f"Forecast_comparison_{index}.png", dpi=300, bbox_inches='tight')
    plt.show()

# for index in [8, 104, 153]:
#     # Extract a sample to perform forecasting
#     sample = get_sample_at_index(val_loader, index=index)
#     df, outcomes = sample_to_dataframe(sample)
#     outcome_30min, outcome_24hours, outcome_resus = outcomes.tolist()


#     ## TIMEGPT Forecast here
#     print("### Forecasting using TimeGPT ###")
#     forecast_results = recursive_forecast_timegpt(df, start_index, end_index, step_size)

#     ## FHR Transformer Forect here, autoregressive mode
#     print("### Forecasting using FHR Tranformer (trained with simple loss) ###")
#     forecast_results_fhrtransformer = recursive_forecast_fhrtransformer(df, start_index, end_index, step_size, model_sim, device)

#      ## FHR Transformer Forect here, autoregressive mode
#     print("### Forecasting using FHR Tranformer (trained with hybrid loss) ###")
#     forecast_results_fhrtransformer_hybrid = recursive_forecast_fhrtransformer(df, start_index, end_index, step_size, model_hybrid, device)


#     # Plot true values, forecasts, and variance, here
#     plt.figure(figsize=(20, 10))
#     data_length = len(df['value'])
#     x_axis = np.arange(-data_length+1, 1)
#     plt.plot(x_axis, df['value'], label='True Values', color='blue', linewidth=3)

#     forecast_length = len(forecast_results['forecast'])
#     forecast_x = np.arange(-forecast_length+1, 1)
#     plt.plot(forecast_x, forecast_results['forecast'], label='TimeGPT Forecast', color='red', linewidth=2, alpha=0.6)
#     rolling_variance_timegpt = forecast_results['forecast'].rolling(window=step_size).var()
#     confidence_interval = 1.96 * np.sqrt(rolling_variance_timegpt)
#     upper_bound = forecast_results['forecast'] + confidence_interval
#     lower_bound = forecast_results['forecast'] - confidence_interval
#     plt.fill_between(forecast_x, lower_bound, upper_bound, color='red', alpha=0.2, label='95% Confidence Interval')


#     plt.plot(forecast_x, forecast_results_fhrtransformer['forecast'], label='FetalFormer Forecast (Simple)', color='green', linewidth=1, alpha=0.8)
#     rolling_variance_fhr = forecast_results_fhrtransformer['forecast'].rolling(window=step_size).var()
#     confidence_interval = 1.96 * np.sqrt(rolling_variance_fhr)
#     upper_bound = forecast_results_fhrtransformer['forecast'] + confidence_interval
#     lower_bound = forecast_results_fhrtransformer['forecast'] - confidence_interval

#     # plt.fill_between(forecast_x, lower_bound, upper_bound, 
#     #                  color='green', alpha=0.2, label='95% Confidence Interval')

#     plt.plot(forecast_x, forecast_results_fhrtransformer_hybrid['forecast'], label='FetalFormer Forecast (Hybrid)', color='y', linewidth=1, alpha=0.6)
#     rolling_variance_fhr_hybrid = forecast_results_fhrtransformer_hybrid['forecast'].rolling(window=step_size).var()
#     confidence_interval = 1.96 * np.sqrt(rolling_variance_fhr_hybrid)
#     upper_bound = forecast_results_fhrtransformer_hybrid['forecast'] + confidence_interval
#     lower_bound = forecast_results_fhrtransformer_hybrid['forecast'] - confidence_interval

#     # plt.fill_between(forecast_x, lower_bound, upper_bound, color='y', alpha=0.2, label='95% Confidence Interval')


#     plt.axvline(x=-3600, color='black', linestyle='--', linewidth=1)

#     plt.ylim(-0.2, 1.2)
#     plt.title(f"True Values and Forecast Approaching the Event\n"
#               f"Outcomes: 30min={outcome_dic[outcome_30min]}, 24h={outcome_dic[outcome_24hours]}, Resus={outcome_resus}", fontsize=28)

#     plt.xlabel('Time to event (seconds)', fontsize=18)
#     plt.ylabel('Value', fontsize=18)
#     plt.legend(fontsize=16)
#     plt.grid(True, linestyle='--', alpha=0.7)

#     plt.tight_layout()
#     plt.savefig(f"TimeGPT_forecast_combined_{index}.png", dpi=300, bbox_inches='tight')
#     plt.show()
