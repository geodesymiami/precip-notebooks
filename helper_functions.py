import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm
from lifelines import CoxTimeVaryingFitter

# Scatter plot of 90 day satellite rain at lat/lon site vs Ayora or Bellavista gauge rain data
def scatter_compare(rainfall, pick, compare_site, site_name, roll_count):
    """ Creates a scatter plot. 
        x-value in graph = Gauge rolling rain measurement at gauge site.
        y-value in graph = Satellite rolling rain measurement at lat/lon site.

    Args:
        rainfall: Satellite rain dataframe for volcanos in chosen region. 
        pick: A site from a dictionary of sites (eg. sites_dict = {'Wolf': (-91.35, .05, 'Wolf'), 'Fernandina': (-91.45, -.45, 'Fernandina')})
        compare_site: Dataframe obtained from (https://www.darwinfoundation.org/en/datazone/climate/puerto-ayora, or https://www.darwinfoundation.org/en/datazone/climate/bellavista)
        site_name: Either 'Ayora' or 'Bellavista'.
        roll_count: Number of days to average rain over.

    Return:
        cleaned_oni: A new dataframe with dates and daily sea surface temperature values.

    """

    # Cleans and aligns dates from gauge and satellite dataframes
    site = data_cleaner(compare_site, roll_count) 

    compare_frame = rainfall.merge(site, on='Date', how='inner')

    # Plots the data
    plt.figure(figsize=(9,5))

    plt.scatter(compare_frame['roll_two'], compare_frame['roll'], color ='maroon')

    plt.xlabel(site_name + ' ' + str(roll_count) + " day gauge rain average (mm)") 
    plt.ylabel(str(pick) + ' ' + str(roll_count) + " day satellite rain average (mm)") 
    plt.title('Plot of rain at ' + site_name + ' against rain at ' + str(pick)) 
    # Data plot
    plt.show()  

    return

# Cleans the Ayora and Bellavista data for the regression
def data_cleaner(dataframe, roll_count):

    frame = dataframe.sort_values(by=['observation_date']).copy()
    frame['Date'] = frame['observation_date']
    frame['roll_two'] = frame.precipitation.rolling(roll_count).mean()
    frame.dropna()

    return frame

# Performs a linear regression on rolling rainfall at two locations
def regressor(rainfall, compare_site, print_summary):
    """ 
    Args:
        rainfall: Rain dataframe that has been pre-processed with volcano_rain_frame function.
        compare_site: Dataframe obtained from (https://www.darwinfoundation.org/en/datazone/climate/puerto-ayora, or https://www.darwinfoundation.org/en/datazone/climate/bellavista).
        print_summary: Boolean variable for if summary is printed.

    Return:
        model_sm: The fitted model.

    """

    # Aligns dates from gauge and satellite dataframes
    compare_frame = rainfall.merge(compare_site, on='Date', how='inner')

    # Runs regression using the statsmodels package
    X_constants = sm.add_constant(compare_frame['roll'])
    model_sm = sm.OLS(compare_frame['roll_two'], X_constants).fit()
    if print_summary == True:
        print(model_sm.summary())

    return model_sm

# Applies linear regression to generate a dataframe of predicted rainfall values
def rain_predictor(rainfall, volcanos, compare_site, roll_count, print_summary):
    """ Creates a new dataframe with rainfall at volcano sites given by feeding gauge data into linear regression model. 

    Args:
        rainfall: Satellite rain dataframe for volcanos in chosen region. 
        volcanos: A dictionary of sites (eg. sites_dict = {'Wolf': (-91.35, .05, 'Wolf'), 'Fernandina': (-91.45, -.45, 'Fernandina')}).
        compare_site: Dataframe obtained from (https://www.darwinfoundation.org/en/datazone/climate/puerto-ayora, or https://www.darwinfoundation.org/en/datazone/climate/bellavista)
        roll_count: Number of days to average rain over.
        print_summary: Boolean variable for if summary is printed.

    Returns:
        pred_rain: A new dataframe with rainfall data given by gauge predictions.

    """

    pred_rain = pd.DataFrame()
    site = data_cleaner(compare_site, roll_count)

    # Performs a linear regression between the gauge site and each volcano individually.
    for pick in volcanos:

        rain_frame = volcano_rain_frame(rainfall, volcanos, pick, roll_count)
        model = regressor(rain_frame, volcanos, pick, site, roll_count, print_summary)
        coefficients = model.params
        coef = coefficients.iloc[1]
        intercept = coefficients.iloc[0]

        longs = [volcanos[pick][0] for i in range(site.shape[0])]
        lats = [volcanos[pick][1] for i in range(site.shape[0])]
        precips = site['precipitation'].apply(lambda x: (coef * x) + intercept)

        volc_rain = pd.DataFrame({'Date': site['Date'], 'Longitude': longs, 'Latitude': lats, 'Precipitation': precips})

        pred_rain = pd.concat([pred_rain, volc_rain], ignore_index=True)

    return pred_rain

# Function used to convert date strings into floats
def date_to_decimal_year(date_str):
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    year = date_obj.year
    day_of_year = date_obj.timetuple().tm_yday
    decimal_year = year + (day_of_year - 1) / 365.0
    decimal_year = round(decimal_year,4) 
    return decimal_year

# Picks out volcano specific rain and adds decimal column, rolling column, and cumsum column
def volcano_rain_frame(rainfall, volcanos, pick, roll_count):
    """ Uses lat/lon, date, and rainfall amount to create a new dataframe that includes site specific decimal dates, rolling average rain, and cumulative rain.

    Args:
        rainfall: Satellite rain dataframe for volcanos in chosen region. 
        volcanos: A dictionary of sites (eg. sites_dict = {'Wolf': (-91.35, .05, 'Wolf'), 'Fernandina': (-91.45, -.45, 'Fernandina')}).
        pick: volcano or site at which to collect data.  
        roll_count: Number of days to average rain over.

    Return:
        volc_rain: A new dataframe with additional columns for decimal date, rolling average, and cumulative rain.

    """    

    # Would be useful if we decide to average over nearby coordinates.
    # lat = volcanos[pick][1]
    # lon = volcanos[pick][0]
    # nearby_rain = rainfall[(abs(lon - rainfall['Longitude']) <= lon_range) & (abs(lat - rainfall['Latitude']) <= lat_range)].copy()
    # dates = np.sort(nearby_rain['Date'].unique())
    # averages = [[date, nearby_rain['Precipitation'][nearby_rain['Date'] == date].mean()] for date in dates]
    # volc_rain = pd.DataFrame(averages, columns = ['Date', 'Precipitation'])

    volc_rain = rainfall[(rainfall['Longitude'] == volcanos[pick][0]) & (rainfall['Latitude'] == volcanos[pick][1])].copy()
    volc_rain['Decimal'] = volc_rain.Date.apply(date_to_decimal_year)
    volc_rain = volc_rain.sort_values(by=['Decimal'])
    volc_rain['roll'] = volc_rain.Precipitation.rolling(roll_count).mean()
    volc_rain = volc_rain.dropna()
    volc_rain['cumsum'] = volc_rain.Precipitation.cumsum()
    return volc_rain

# Picks out all eruptions of a specific volcano within a certain date range.
def volcano_erupt_dates(eruptions, pick, period_start, period_end):
    """ Picks out all eruptions of a specific volcano within a certain date range.

    Args:
        eruptions: A dataframe with columns-- 'Volcano' and 'Start'. 'Start' is the beginning date of the eruption given as a string-- YYYY-MM-DD.
        pick: volcano or site at which to collect data.  
        period_start: Beginning of date range.
        period_end: End of date range.

    Return:
        erupt_dates: Array of decimal dates for eruptions.

    """  
    volc_erupts = eruptions[eruptions['Volcano'] == pick].copy()
    volc_erupts['Decimal'] = volc_erupts.Start.apply(date_to_decimal_year)
    erupt_dates = np.array(volc_erupts['Decimal'][(volc_erupts['Decimal'] >= period_start) & (volc_erupts['Decimal'] <= period_end)])
    return erupt_dates

### STILL UNDER CONSTRUCTION ###
# Performs a cox regression for a chosen volcano
def cox_regressor(rainfall, eruptions, volcanos, roll_count, lower_cutoff, upper_cutoff, shift):

    list_volcs = list(volcanos.keys())
    cox = pd.DataFrame()

    for pick in list_volcs:

        volc_rain = volcano_rain_frame(rainfall, volcanos, pick, roll_count)
        volc_rain['roll'] = volc_rain['roll'].apply(lambda x: max(x-lower_cutoff, 0))
        volc_rain['roll'] = volc_rain['roll'].apply(lambda x: min(upper_cutoff, x))
        volc_rain['roll'] = volc_rain['roll'].shift(shift) 
        
        starts = np.array(eruptions['Start'][eruptions['Volcano'] == pick])
        for i in range(len(starts)):
            volc_dict = {}
            if i == len(starts) - 1:
                erupt_interval = volc_rain[(volc_rain['Date'] >= starts[i])].sort_values(by='Date')
                event = [0 for i in range(len(erupt_interval)-1)] + [0]
            else:
                erupt_interval = volc_rain[(volc_rain['Date'] >= starts[i]) & (volc_rain['Date'] < starts[i+1])].sort_values(by='Date')
                event = [0 for k in range(len(erupt_interval)-1)] + [1]
            for k in list_volcs:
                if k == pick:
                    volc_dict[list_volcs.index(k)] = [1 for l in range(len(erupt_interval))]
                else:
                    volc_dict[list_volcs.index(k)] = [0 for l in range(len(erupt_interval))]

            date = date_to_decimal_year(starts[i])
            
            birth = [date for k in range(len(erupt_interval))]
            start = [k for k in range(len(erupt_interval))]
            stop = [k+1 for k in range(len(erupt_interval))]

            data = list(zip(birth, start, stop, list(erupt_interval['roll']), volc_dict[0], volc_dict[1], volc_dict[2], volc_dict[3], event))
            newborn = pd.DataFrame(data, columns=['Birth', 'Start Day', 'Stop Day', 'Precipitation', 'Cerro Azul', 'Fernandina', 'Sierra Negra', 'Wolf', 'Death'])
            cox = pd.concat([cox, newborn], ignore_index=True)
    ctv = CoxTimeVaryingFitter(penalizer=0.0000001)
    ctv.fit(cox, id_col='Birth', event_col='Death', start_col='Start Day', stop_col='Stop Day')
    ctv.print_summary()  

    return  