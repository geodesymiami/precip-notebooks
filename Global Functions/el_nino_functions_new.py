import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from helper_functions import volcano_rain_frame, volcano_erupt_dates, date_to_decimal_year, recurrences, create_map, color_scheme, quantile_name
from scipy.stats import gamma
import statsmodels.api as sm
import os
import sys

oni = pd.read_csv(os.path.join('/Users/jonathanquartin/Documents/Coding/GitHub/precip-notebooks/GALAPAGOS_DATA', "oni_2024.csv"))
eruptions = pd.read_csv(os.path.join('/Users/jonathanquartin/Documents/Coding/GitHub/precip-notebooks/GALAPAGOS_DATA', "erupt_2024.csv"))

def nino_by_strength(lon, lat, start_date, end_date, folder, roll_count=1):
    """ Sorts days by rain value (smallest to largest) and generates a histogram of where elnino days fall.

    Args:
        lon: The longitude of the location of interest (rounded to the nearest .05)
        lat: The latitude of the location of interest (rounded to the nearest .05)
        start_date: String YYYY-MM-DD for the first date of interest
        end_date: String YYYY-MM-DD for the last date of interest
        folder: Location where the rain data is stored
        roll_count: Number of days to average rain over

    Return:
    """

    global oni

    rainfall = create_map(lat, lon, [start_date, end_date], folder)

    elninos = nino_dict(oni, rainfall)

    new_rain = volcano_rain_frame(rainfall, roll_count, lon, lat)

    sorted_rain = new_rain.sort_values(by=['roll'])
    dates = np.array(sorted_rain['Decimal'])

    bin_count = 60
    bin_length = len(dates) / bin_count
    bins = [0 for i in range(bin_count)]

    for i in range(len(dates)):
        for j in elninos['moderate nino']:
            if dates[i] >= j[0] and dates[i] <= j[1]:
                bins[int(i // bin_length)] += 1
                break

    # Create bar plot
    plt.figure(figsize=(12,6))
    plt.bar([i for i in range(bin_count)], bins, color='red')

    # Adding labels and title
    plt.xlabel('Sorted days by rain (made into 60 bins)')
    plt.ylabel('Count of El Niño days')
    plt.title('El Niño by rain amount')

    # Display the plot
    plt.show()

    return

def oni_type_precip(lon, lat, start_date, end_date, folder, roll_count=1):
    """ Generates an ONI-type rainfall plot (centered around average), and also a plot of ONI data (for comparison).

    Args:
        lon: The longitude of the location of interest (rounded to the nearest .05)
        lat: The latitude of the location of interest (rounded to the nearest .05)
        start_date: String YYYY-MM-DD for the first date of interest
        end_date: String YYYY-MM-DD for the last date of interest
        folder: Location where the rain data is stored
        roll_count: Number of days to average rain over

    Return:
    """

    global oni

    rainfall = create_map(lat, lon, [start_date, end_date], folder)
    start = int(start_date[0:4])
    end = int(end_date[0:4])+1

    nino_compare_frame = nino_rain_compare(oni, rainfall, roll_count)
    degree_symbol = '\u00b0'
    temperature = 25

    # Sample data
    categories = np.array(nino_compare_frame['Decimal'])
    values1 = np.array(nino_compare_frame['roll'])
    values2 = np.array(nino_compare_frame['ANOM'])

    colors1 = ['red' if val >= 0 else 'blue' for val in values1]
    colors2 = ['red' if val >= 0 else 'blue' for val in values2]
    # Create bar plot

    fig, axs = plt.subplots(2, 1, figsize=(15, 12))

    axs[0].bar(categories, values1, color=colors1, width=0.2)

    # Add title and labels
    axs[0].set_title('ONI-type precipitation plot (Isabela)')
    axs[0].set_xticks([start + (i) for i in range(end-start+1)], ["'" + str(start + (i))[2:] for i in range(end-start+1)])
    axs[0].set_xlabel('Years')
    axs[0].set_ylabel('Rain index (mm)')

    axs[1].bar(categories, values2, color=colors2, width=0.2)

    # Add title and labels
    axs[1].set_title('ONI')
    axs[1].set_xticks([start + (i) for i in range(end-start+1)], ["'" + str(start + (i))[2:] for i in range(end-start+1)])
    axs[1].set_xlabel('Years')
    axs[1].set_ylabel('Oceanic Niño Index (' + f"{degree_symbol}C" + ')')

    # Show plot
    plt.show()

    return


def nino_scatter(lon, lat, start_date, end_date, folder, roll_count=1):
    """ Plots (x,y) coordinates for days where x is ONI ANOM value, and y is the precipitation value.

    Args:
        lon: The longitude of the location of interest (rounded to the nearest .05)
        lat: The latitude of the location of interest (rounded to the nearest .05)
        start_date: String YYYY-MM-DD for the first date of interest
        end_date: String YYYY-MM-DD for the last date of interest
        folder: Location where the rain data is stored
        roll_count: Number of days to average rain over

    Return:
    """

    global oni

    rainfall = create_map(lat, lon, [start_date, end_date], folder)

    nino_compare_frame = nino_rain_compare(oni, rainfall, roll_count)
    degree_symbol = '\u00b0'
    colors = plt.cm.plasma(np.linspace(.95, .5, len(nino_compare_frame)))

    plt.figure(figsize=(12,6))

    anoms = np.array(nino_compare_frame['ANOM'])
    rolls = np.array(nino_compare_frame['roll'])

    for i in range(len(nino_compare_frame)):

        plt.scatter(anoms[i], rolls[i], color= colors[i])

    plt.ylabel('Centered 90-day precipitation (mm)')
    plt.xlabel('Oceanic Niño Index (' + f"{degree_symbol}C" + ')')
    plt.title('Rain versus ONI')

    plt.show()

    return


def nino_distribution(lon, lat, start_date, end_date, folder, roll_count=1, cutoff='weak nino'):
    """ Two sub-histograms of nino (and non-nino) days, by rain amount. Also plots gamma distributions to best fit the data.

    Args:
        lon: The longitude of the location of interest (rounded to the nearest .05)
        lat: The latitude of the location of interest (rounded to the nearest .05)
        start_date: String YYYY-MM-DD for the first date of interest
        end_date: String YYYY-MM-DD for the last date of interest
        folder: Location where the rain data is stored
        roll_count: Number of days to average rain over
        cutoff: weak nino, moderate nino, strong nino, very strong nino

    Return:
    """
    global oni

    rainfall = create_map(lat, lon, [start_date, end_date], folder)

    elninos = nino_dict(oni, rainfall)

    merg, ninos = nino_separator(rainfall, elninos, cutoff, roll_count)
    frames = [merg, ninos]
    fig, axes = plt.subplots(2, 1, figsize=(12,12))

    for i in range(2):
        mean = frames[i]['roll'].mean()
        std_dev = np.std(frames[i]['roll'])

        sorted_merg = frames[i].sort_values(by=['roll'])
        vals = sorted_merg['roll'].apply(lambda x: max(x,0))
        vals = vals.dropna()

        # Calculate alpha and beta from mean and standard deviation
        beta = (std_dev ** 2) / mean
        alpha = mean / beta

        # Generate data points for the x-axis
        x = np.linspace(0, 800, 1000)

        # Calculate the corresponding probability density function (PDF) values
        pdf = gamma.pdf(x, alpha, scale=beta)

        # Plot scaled histogram
        axes[i].hist(vals, bins=100, density=True, label='Normalized 90-day rain distribution')
        axes[i].plot(x, pdf, color='red', label='Gamma distribution')
        axes[i].axvline(mean, color='black', linestyle='dashed', linewidth=1, label = 'Mean')

        # Show plot
        axes[i].grid(True)

        # Get the current x-tick locations and labels
        xticks = axes[i].get_xticks()

        # Append the new tick location to the list of ticks
        new_xticks = list(xticks) + [mean]

        axes[i].set_xticks(new_xticks)
        axes[i].set_xlabel('90-day precipitation')
        axes[i].set_ylabel('Normalized day count')
        axes[i].set_ylim(0,.065)
        axes[i].set_xlim(0,500)
        if i == 0:
            axes[i].set_title('Distribution of rain in Non-El Niño Days')
        else:
            axes[i].set_title('Distribution of rain in El Niño Days')

        # Add legend
        axes[i].legend()

    plt.show()

    return merg, ninos

def nino_separator(rainfall, elninos, roll_count, cutoff):
    """ Separates out the rainfall data associated with days that fall within El Nino periods.

    Args:
        rainfall: Built from the create_map function
        lat: Built from the nino_dict function
        roll_count: Number of days to average rain over
        cutoff: weak nino, moderate nino, strong nino, very strong nino

    Return:
        merg: A subframe of 'rainfall' consisting of data from days that don't fall in El Nino periods.
        ninos: A subframe of 'rainfall' consisting of data from days that fall in El Nino periods.
    """
    
    merg = volcano_rain_frame(rainfall, roll_count)
    ninos = pd.DataFrame()

    for i in elninos[cutoff]:
        ninos = pd.concat([ninos, merg[(merg['Decimal'] <= i[1]) & (merg['Decimal'] >= i[0])]])
        merg = merg[~((merg['Decimal'] <= i[1]) & (merg['Decimal'] >= i[0]))]

    return merg, ninos

def nina_separator(rainfall, elninos, cutoff, roll_count):
    """ Separates out the rainfall data associated with days that fall within La Nina periods.

    Args:
        rainfall: Built from the create_map function
        lat: Built from the nino_dict function
        roll_count: Number of days to average rain over
        cutoff: weak nina, moderate nina, strong nina

    Return:
        merg: A subframe of 'rainfall' consisting of data from days that don't fall in La Nina periods.
        ninas: A subframe of 'rainfall' consisting of data from days that fall in La Nina periods.
    """

    merg = volcano_rain_frame(rainfall, roll_count)
    ninas = pd.DataFrame()

    for i in elninos[cutoff]:
        ninas = pd.concat([ninas, merg[(merg['Decimal'] <= i[1]) & (merg['Decimal'] >= i[0])]])
        merg = merg[~((merg['Decimal'] <= i[1]) & (merg['Decimal'] >= i[0]))]

    return merg, ninas

def nino_rain_compare(oni, rainfall, roll_count):
    """ Aligns oni data with rainfall data, by day.

    Args:
        oni: global variable-- oni dataframe from https://www.climate.gov/news-features/understanding-climate/climate-variability-oceanic-nino-index
        rainfall: Built from the create_map function
        roll_count: Number of days to average rain over

    Return:
        merged_df: A data frame consisting of the combined oni and rainfall data.
    """

    new_rain = volcano_rain_frame(rainfall, roll_count, None, None, True, False)
    average = new_rain['Precipitation'].mean()
    new_rain['roll'] = new_rain['roll'].apply(lambda x: (x/roll_count) - average)

    new_nino = elnino_cleaner()

    merged_df = pd.merge(new_rain, new_nino, left_on='Date', right_on='Center', how='inner')  

    return merged_df  

def nino_dict(oni, rainfall):
    """ Creates a dictionary of el nino and la nina events within the range of the rainfall time period.

    Args:
        oni: global variable-- oni dataframe from https://www.climate.gov/news-features/understanding-climate/climate-variability-oceanic-nino-index
        rainfall: Built from the create_map function

    Return:
        elninos: {'weak ninos':[[start_event, end_event], ...], 'weak ninas':[], ...}
    """

    start = min(rainfall['Date'])
    end = max(rainfall['Date'])
    strengths = {}
    anom_types = {'weak nina': -.5, 'moderate nina': -1, 'strong nina': -1.5, 'weak nino': .5, 'moderate nino': 1, 'strong nino': 1.5, 'very strong nino': 2}
    for i in anom_types:
        if i == 'weak nina' or i == 'weak nino':
            strengths.update(elnino_strengths(anom_types[i], i, 5))
        else:
            strengths.update(elnino_strengths(anom_types[i], i, 3)) 

    # Picks out elninos/laninas between the start and end dates. Converts dates to decimals.
    elninos = {'weak nina': [], 'moderate nina': [], 'strong nina': [], 'weak nino': [], 'moderate nino': [], 'strong nino': [], 'very strong nino': []}

    if strengths != None:
        for j in strengths:
            for i in strengths[j]:
                if i[1] > start and i[0] < end:
                    first = max(i[0], start)
                    last = min(i[1], end)
                    elninos[j].append([date_to_decimal_year(first), date_to_decimal_year(last)])

    return elninos

# Strength finder
def elnino_strengths(val, strength, cutoff):
    """ Determines the date ranges for El Nino or La Nina events of a specified strength.

    Typical classification of nino/nina events:
    anom_types = {'weak nina': -.5, 'moderate nina': -1, 'strong nina': -1.5, 'weak nino': .5, 'moderate nino': 1, 'strong nino': 1.5, 'very strong nino': 2}

    Args:
        val: Chosen cutoff sea surface temp for an El Nino event (eg. > 0.5 signifies all El Nino events).
        strength: Intensity of the event type considered (eg. for > 2, strength would be the string "very strong el nino")
        cutoff: Typically 0.5 or -0.5 indicating being within any type of Nino or Nina event.

    Return:
        period: A one key dictionary with key given by the strength, and value given by a list of pairs [start, end],
        where start is the beginning of an event, and end is the conclusion of an event.

    """

    global oni

    # Add a new column to ONI data for the 15th of the middle month. eg. For 'DJF' of 2000, the new column entry is '2000-01-15'
    date_converter = {'DJF': '01', 'JFM': '02', 'FMA': '03', 'MAM': '04', 'AMJ': '05', 'MJJ': '06', 'JJA': '07', 'JAS': '08', 'ASO': '09', 'SON': '10', 'OND': '11', 'NDJ': '12'}
    
    def convert_mid(row):
        return str(row['YR']) + '-' + date_converter[row['SEAS']] + '-15' 
    
    orig = oni.copy()
    orig['Center'] = oni.apply(convert_mid, axis=1)

    # Algorithm for finding date ranges. Checks if 5 consecutive rows meet the criteria for an event (same criterion as used on ONI website)
    period = {strength: []}
    oni_array = np.array(orig)
    count = 0
    while count < (oni_array.shape[0] - 5):
        if val > 0:
            if oni_array[count][3] >= val:
                first = count
                event = True
                for j in range(cutoff-1):
                    count += 1
                    if count == oni_array.shape[0]:
                        break
                    if oni_array[count][3] < val:
                        event = False
                        break
                if event == True:
                    start = oni_array[first][4]
                    while oni_array[count][3] >= val:
                        count += 1
                        if count == oni_array.shape[0]:
                            break
                    end = oni_array[count-1][4]
                    period[strength].append([start, end])
            else:
                count += 1
        elif val < 0:
            if oni_array[count][3] <= val:
                first = count
                event = True
                for j in range(cutoff-1):
                    count += 1
                    if count == oni_array.shape[0]:
                        break
                    if oni_array[count][3] > val:
                        event = False
                        break
                if event == True:
                    start = oni_array[first][4]
                    while oni_array[count][3] <= val:
                        count += 1
                        if count == oni_array.shape[0]:
                            break
                    end = oni_array[count-1][4]
                    period[strength].append([start, end])
            else:
                count += 1
    
    return period



# El nino day value
def elnino_cleaner():
    """ Creates a copy of the ONI dataframe and adds columns associated to the start, center, and end of a period.

    Args:

    Return:
        orig: The new dataframe

    """
    
    global oni

    # Add a new column for the first day of each month in a string. (eg. for 'DJF' add a column for the 1st day of Dec, Jan, and Feb)
    date_converter = {'DJF': ('12', '03', '01'), 'JFM': ('01', '04', '02'), 'FMA': ('02', '05', '03'), 'MAM': ('03', '06', '04'), 'AMJ': ('04', '07', '05'), 'MJJ': ('05', '08', '06'), 'JJA': ('06', '09', '07'), 'JAS': ('07', '10', '08'), 'ASO': ('08', '11', '09'), 'SON': ('09', '12', '10'), 'OND': ('10', '01', '11'), 'NDJ': ('11', '02', '12')}
    
    def convert_start(row):
        if row['SEAS'] == 'DJF':
            return str(row['YR']-1) + '-' + date_converter[row['SEAS']][0] + '-01'
        else:
            return str(row['YR']) + '-' + date_converter[row['SEAS']][0] + '-01' 

    def convert_end(row):
        if row['SEAS'] == 'NDJ':
            return str(row['YR']+1) + '-' + date_converter[row['SEAS']][1] + '-01'
        else: 
            return str(row['YR']) + '-' + date_converter[row['SEAS']][1] + '-01' 
    
    def convert_mid(row):
        return str(row['YR']) + '-' + date_converter[row['SEAS']][2] + '-15' 
    
    orig = oni.copy()
    orig['Start'] = orig.apply(convert_start, axis=1)
    orig['Center'] = orig.apply(convert_mid, axis=1)
    orig['End'] = orig.apply(convert_end, axis=1) 


    return orig