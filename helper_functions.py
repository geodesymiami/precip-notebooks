import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm
from lifelines import CoxTimeVaryingFitter

# Creates histograms that break up eruption data based on quantile of rainfall.
def mix_counter(volcanos, eruptions, rainfall, roll_count, recur=False):
    """ For each volcano, breaks up rain data by amount, and bins eruptions based on this. Generates histograms for each volcano
    separately, and also a histogram that puts all of the eruption data together.

    Args:
        volcanos: A dictionary of sites (eg. sites_dict = {'Wolf': (-91.35, .05, 'Wolf'), 'Fernandina': (-91.45, -.45, 'Fernandina')}).
        eruptions: A dataframe with columns-- 'Volcano' and 'Start'. 'Start' is the beginning date of the eruption given as a string-- YYYY-MM-DD.
        rainfall: Satellite rain dataframe for volcanos in chosen region. 
        color_count: Number of quantiles to break rain data into.
        roll_count: Number of days to average rain over.
        by_season: Boolean for if quantiles should be made for every year separately, or across the entire date range at once.

    Return:

    """

    # Creates a dictionary where for each volcano, we get an array of eruptions in each quantile.
    totals = {volcano:0 for volcano in volcanos}
    erupt_vals = {pick:[] for pick in totals}
    recurs = recurrences(eruptions, volcanos)
    rain_tots = {pick:[] for pick in totals}


    for pick in totals:

        # Get volcano specific data and order dates by 'roll' amount
        volc_init = volcano_rain_frame(rainfall, volcanos, pick, roll_count)
        start = int(volc_init['Decimal'].min() // 1)
        end = int(volc_init['Decimal'].max() // 1)

        erupt_dates = volcano_erupt_dates(eruptions, pick, start, end)
        if recur == True:
            volc_rain = volc_init.copy()
            for i in range(len(erupt_dates)):
                volc_rain = volc_rain[~((volc_init['Decimal'] > erupt_dates[i]) & (volc_init['Decimal'] < erupt_dates[i] + recurs[pick]))].copy()
                
        else:
            volc_rain = volc_init.copy()
            
        dates = volc_rain.sort_values(by=['roll']).copy()
        dates = dates.dropna()
        date_dec = np.array(dates['Decimal'])
        date_rain = np.array(dates['roll'])
        
        for k in erupt_dates:
            for i in range(len(date_dec)):
                if k == date_dec[i]:
                    erupt_vals[pick].append(date_rain[i])

        rain_by_year = []
        for k in range(start, end + 1):
            rain_by_year.append(volc_rain['Precipitation'][(volc_rain['Date'] >= str(k)) & (volc_rain['Date'] < str(k+1))].sum()) 
        mean = sum(rain_by_year) / (len(rain_by_year))
        
        count=0
        for j in range(start, end + 1):
            length = ((0*mean) + (50*rain_by_year[count])) / ((100*mean) + (100*rain_by_year[count])) 
            dates_j = np.array([day for day in date_dec if (day // 1) == j])
            bin_size = int((length * len(dates_j)) // 1)
            rain_tots[pick].append(bin_size)
            quantile = dates_j[-(bin_size):]
            for k in erupt_dates:
                if k in quantile:
                    totals[pick] += 1  
                    erupt_vals[pick].append(k) 
            count+=1

    return totals, erupt_vals, rain_tots

# Computes the recurrence times for volcanoes
def recurrences(volcanic_events, sites_dict, duration=.75):
    eruptions = volcanic_events.copy()
    eruptions['Decimal'] = eruptions.Start.apply(date_to_decimal_year)
    recurrences = {}
    for i in sites_dict:

        erupt_dates = list(eruptions['Decimal'][eruptions['Volcano'] == i])
        recurrences[i] = duration

    return recurrences


# Scatter plot of 90 day satellite rain at lat/lon site vs Ayora or Bellavista gauge rain data
def scatter_compare(rainfall, sites_dict, gauges_dict, compare_site, site_name, roll_count):
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

    fig, axes = plt.subplots((len(sites_dict) // 2) + 1, 2, figsize=(10, 15))

    rain_frame = volcano_rain_frame(rainfall, gauges_dict, site_name, roll_count)
    # Cleans and aligns dates from gauge and satellite dataframes
    site = data_cleaner(compare_site, roll_count) 

    compare_frame = rain_frame.merge(site, on='Date', how='inner')

    # Plots the data

    axes[0, 0].scatter(compare_frame['roll_two'], compare_frame['roll'], color ='maroon')
    model = regressor(rain_frame, compare_site, True)
    coefficients = model.params
    coef = coefficients.iloc[1]
    intercept = coefficients.iloc[0]
    axes[0, 0].plot(compare_frame['roll_two'], compare_frame['roll_two'].apply(lambda x: (coef * x) + intercept), color ='black', alpha=1.0, linewidth=3)
    axes[0, 0].set_xlabel(site_name + ' ' + str(roll_count) + " day gauge rain average (mm)") 
    axes[0, 0].set_ylabel(str(site_name) + ' ' + str(roll_count) + " day satellite rain average (mm)") 
    axes[0, 0].set_title(site_name + ' gauge vs ' + str(site_name) + ' satellite') 

    fig.delaxes(axes[0, 1])

    count = 0
    for pick in sites_dict:

        rain_frame = volcano_rain_frame(rainfall, sites_dict, pick, roll_count)
    
        # Cleans and aligns dates from gauge and satellite dataframes
        site = data_cleaner(compare_site, roll_count) 

        compare_frame = rain_frame.merge(site, on='Date', how='inner')

        # Plots the data

        axes[(count // 2) + 1, count % 2].scatter(compare_frame['roll_two'], compare_frame['roll'], color ='maroon')
        print(str(pick))
        model = regressor(rain_frame, compare_site, True)
        coefficients = model.params
        coef = coefficients.iloc[1]
        intercept = coefficients.iloc[0]
        axes[(count // 2) + 1, count % 2].plot(compare_frame['roll_two'], compare_frame['roll_two'].apply(lambda x: (coef * x) + intercept), color ='black', alpha=1.0, linewidth=3)
        axes[(count // 2) + 1, count % 2].set_xlabel(site_name + ' ' + str(roll_count) + " day gauge rain average (mm)") 
        axes[(count // 2) + 1, count % 2].set_ylabel(str(sites_dict[pick][2]) + ' ' + str(roll_count) + " day satellite rain average (mm)") 
        axes[(count // 2) + 1, count % 2].set_title(site_name + ' gauge vs ' + str(sites_dict[pick][2]) + ' satellite') 

        count += 1

    return

# Cleans the Ayora and Bellavista data for the regression
def data_cleaner(dataframe, roll_count):

    frame = dataframe.sort_values(by=['observation_date']).copy()
    frame['Date'] = frame['observation_date']
    frame['roll_two'] = frame.precipitation.rolling(roll_count).mean()
    frame.dropna()

    return frame

# Performs a linear regression on rolling rainfall at two locations
def regressor(rainfall, compare_site, print_summary=False):
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
    X_constants = sm.add_constant(compare_frame['roll_two'])
    model_sm = sm.OLS(compare_frame['roll'], X_constants).fit()
    if print_summary == True:
        print(model_sm.summary())

    return model_sm

# Applies linear regression to generate a dataframe of predicted rainfall values
def rain_predictor(rainfall, volcanos, compare_site, roll_count, print_summary=False):
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
        model = regressor(rain_frame, site, print_summary)
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