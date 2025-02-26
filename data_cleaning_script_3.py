import pandas as pd
import numpy as np

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
    
    # Find and replace the highest Recovered value for each country
    max_recovered_per_country = df_processed.groupby('Country')['Recovered'].transform('max')
    
    # Replace the highest Recovered value with 0
    df_processed.loc[df_processed['Recovered'] == max_recovered_per_country, 'Recovered'] = 0
    
    # Handle the Recovered column imputation - simple approach
    countries = df_processed['Country'].unique()
    
    for country in countries:
        country_mask = df_processed['Country'] == country
        country_data = df_processed[country_mask]
        
        # Find where recovery data becomes and stays zero
        non_zero_indices = country_data.index[country_data['Recovered'] > 0].tolist()
        
        if not non_zero_indices:
            continue  # No recovery data for this country
            
        # Find the last non-zero recovery value
        last_non_zero_idx = non_zero_indices[-1]
        
        # Check if there are entries after this point
        later_indices = country_data.index[country_data.index > last_non_zero_idx].tolist()
        
        # If all later values are zero, we need to impute
        if later_indices and all(country_data.loc[later_indices, 'Recovered'] == 0):
            # Get the valid data before the cutoff
            valid_data = country_data.loc[non_zero_indices]
            
            # Calculate the mean and standard deviation of the daily recovered values
            mean_recoveries = valid_data['Recovered'].mean()

            std_recoveries = valid_data['Recovered'].std()
            
            # If std is NaN or 0, use a percentage of the mean
            if np.isnan(std_recoveries) or std_recoveries == 0:
                std_recoveries = mean_recoveries * 0.3  # 30% of mean as a reasonable variation
                
            # Generate random values based on the distribution of previous data
            for idx in later_indices:
                # Create a random value with similar characteristics to the previous data
                random_recovery = np.random.normal(mean_recoveries, std_recoveries)
                # Ensure it's positive and an integer
                random_recovery = max(0, round(random_recovery))
                df_processed.loc[idx, 'Recovered'] = random_recovery
    
    return df_processed

df = pd.read_csv('dataset/new_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])
df_processed = process_covid_data(df)
df_processed.to_csv('file.csv', index=False)