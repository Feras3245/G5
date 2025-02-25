import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Function to convert cumulative data to daily and handle the Recovered column imputation
def process_covid_data(df):
    # Make a copy of the original DataFrame
    df_processed = df.copy()
    
    # Ensure data is sorted by Country and Date
    df_processed = df_processed.sort_values(['Country', 'Date'])
    
    # Convert cumulative to daily for all three columns
    for column in ['Confirmed', 'Deaths', 'Recovered']:
        df_processed[column] = df_processed.groupby('Country')[column].diff().fillna(df_processed[column])
        # Convert negative values to absolute values
        df_processed[column] = df_processed[column].abs()
    
    # Handle the Recovered column imputation
    countries = df_processed['Country'].unique()
    
    for country in countries:
        country_data = df_processed[df_processed['Country'] == country].copy()
        
        # Check if there's a reset to zero after some non-zero values
        has_recovery_data = country_data['Recovered'].any()
        if not has_recovery_data:
            continue  # Skip if no recovery data at all
        
        # Find where recovery data resets to zero (and stays zero)
        recovery_indices = country_data.index[country_data['Recovered'] > 0].tolist()
        if not recovery_indices:
            continue
        
        # If there are recovery values, then zeros, then more recovery values, this isn't a permanent reset
        last_recovery_idx = recovery_indices[-1]
        remaining_indices = country_data.index[country_data.index > last_recovery_idx].tolist()
        
        # Check if there's a permanent reset (all zeros after last recovery)
        all_zeros_after = all(country_data.loc[remaining_indices, 'Recovered'] == 0) if remaining_indices else False
        
        # Only process countries with the reset pattern
        if all_zeros_after and remaining_indices:
            # Get data before the reset to base the imputation on
            valid_data = country_data.loc[recovery_indices]
            
            # Calculate recovery rate as a percentage of confirmed cases
            if len(valid_data) >= 14:  # Ensure enough data for calculation
                # Use last 14 days for a stable calculation
                recent_data = valid_data.iloc[-14:]
                recovery_rate = recent_data['Recovered'].sum() / recent_data['Confirmed'].sum() if recent_data['Confirmed'].sum() > 0 else 0
            else:
                recovery_rate = valid_data['Recovered'].sum() / valid_data['Confirmed'].sum() if valid_data['Confirmed'].sum() > 0 else 0
            
            # If we can't calculate a rate, use a reasonable default
            if np.isnan(recovery_rate) or recovery_rate == 0:
                recovery_rate = 0.85  # Assume 85% recovery rate as a default
            
            # Add some randomness to the recovery rate (±10%)
            for idx in remaining_indices:
                confirmed_cases = country_data.loc[idx, 'Confirmed']
                # Calculate base recovery with a lag of ~14 days (recoveries lag behind confirmations)
                lagged_idx = idx - 14
                if lagged_idx in country_data.index and country_data.loc[lagged_idx, 'Country'] == country:
                    lagged_confirmed = country_data.loc[lagged_idx, 'Confirmed']
                    # Apply recovery rate with randomness
                    random_factor = np.random.uniform(0.9, 1.1)  # ±10% randomness
                    imputed_recovery = lagged_confirmed * recovery_rate * random_factor
                    df_processed.loc[idx, 'Recovered'] = max(0, round(imputed_recovery))
                
    return df_processed

# Example of how to use the function
# df = pd.read_csv('covid_data.csv')  # Replace with your actual data loading
# df['Date'] = pd.to_datetime(df['Date'])  # Ensure Date is datetime
# df_processed = process_covid_data(df)

# Function to verify imputation with visualization
def plot_country_data(df_original, df_processed, country, start_date=None, end_date=None):
    # Filter data for the specified country
    country_orig = df_original[df_original['Country'] == country].copy()
    country_proc = df_processed[df_processed['Country'] == country].copy()
    
    # Apply date filters if provided
    if start_date:
        country_orig = country_orig[country_orig['Date'] >= start_date]
        country_proc = country_proc[country_proc['Date'] >= start_date]
    if end_date:
        country_orig = country_orig[country_orig['Date'] <= end_date]
        country_proc = country_proc[country_proc['Date'] <= end_date]
    
    # Create plot
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot Confirmed cases
    axs[0].plot(country_orig['Date'], country_orig['Confirmed'], 'b-', label='Original (Cumulative)')
    axs[0].plot(country_proc['Date'], country_proc['Confirmed'], 'r-', label='Processed (Daily)')
    axs[0].set_title(f'{country} - Confirmed Cases')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot Deaths
    axs[1].plot(country_orig['Date'], country_orig['Deaths'], 'b-', label='Original (Cumulative)')
    axs[1].plot(country_proc['Date'], country_proc['Deaths'], 'r-', label='Processed (Daily)')
    axs[1].set_title(f'{country} - Deaths')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot Recovered
    axs[2].plot(country_orig['Date'], country_orig['Recovered'], 'b-', label='Original (Cumulative)')
    axs[2].plot(country_proc['Date'], country_proc['Recovered'], 'r-', label='Processed (Daily/Imputed)')
    axs[2].set_title(f'{country} - Recovered')
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()
    return fig

# Example usage for visualization:
# fig = plot_country_data(df, df_processed, 'Italy')
# plt.show()

dataset = pd.read_csv('dataset/new_dataset.csv')
dataset = process_covid_data(dataset)
dataset.to_csv("cleaned_data_2.csv", index=False)