import pandas as pd

old_dataset = pd.read_csv('dataset/time-series-19-covid-combined.csv')
new_dataset = pd.DataFrame(columns=['Date', 'Country', 'Confirmed', 'Recovered', 'Death'])

countries = old_dataset['Country/Region'].unique()
country_datasets = list()

for country in countries:
    country_dataset = pd.DataFrame(old_dataset[old_dataset['Country/Region'] == country])
    if country_dataset['Province/State'].any():
        country_dataset = country_dataset.groupby('Date', as_index=False)[['Confirmed', 'Recovered', 'Deaths']].sum()
        country_dataset['Country'] = country
        country_dataset = country_dataset.loc[:, ['Date', 'Country', 'Confirmed', 'Recovered', 'Deaths']] 
    else:
        country_dataset = country_dataset.drop(['Province/State'], axis=1)
        country_dataset = country_dataset.rename({'Country/Region':'Country'}, axis=1)
    
    country_dataset = country_dataset.sort_values(by='Date', ascending=True)
    country_datasets.append(country_dataset)


new_dataset = pd.concat(country_datasets, ignore_index=True, sort=False)
    
new_dataset.to_csv('new_dataset.csv', index=False)

