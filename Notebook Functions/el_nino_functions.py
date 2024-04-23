import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from helper_functions import volcano_rain_frame, volcano_erupt_dates, date_to_decimal_year, recurrences
from scipy.stats import gamma
import statsmodels.api as sm

def nino_by_strength(rainfall, elninos, roll_count):

    new_rain = volcano_rain_frame(rainfall, roll_count)

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

def oni_type_precip(oni, rainfall, rolling_number):

    nino_compare_frame = nino_rain_compare(oni, rainfall, rolling_number)
    degree_symbol = '\u00b0'
    temperature = 25
    print(f"{temperature}{degree_symbol}C")

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
    axs[0].set_xticks([1965 + (2*i) for i in range(31)], ["'" + str(1965 + (2*i))[2:] for i in range(31)])
    axs[0].set_xlabel('Years')
    axs[0].set_ylabel('Rain index (mm)')

    axs[1].bar(categories, values2, color=colors2, width=0.2)

    # Add title and labels
    axs[1].set_title('ONI')
    axs[1].set_xticks([1965 + (2*i) for i in range(31)], ["'" + str(1965 + (2*i))[2:] for i in range(31)])
    axs[1].set_xlabel('Years')
    axs[1].set_ylabel('Oceanic Niño Index (' + f"{degree_symbol}C" + ')')

    # Show plot
    plt.show()

    return


def nino_scatter(oni, rainfall, rolling_number):

    nino_compare_frame = nino_rain_compare(oni, rainfall, rolling_number)
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


def nino_distribution(rainfall, elninos, cutoff, roll_count):

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

def nino_separator(rainfall, elninos, cutoff, roll_count):
    
    merg = volcano_rain_frame(rainfall, roll_count)
    ninos = pd.DataFrame()

    for i in elninos[cutoff]:
        ninos = pd.concat([ninos, merg[(merg['Decimal'] <= i[1]) & (merg['Decimal'] >= i[0])]])
        merg = merg[~((merg['Decimal'] <= i[1]) & (merg['Decimal'] >= i[0]))]

    return merg, ninos

def nina_separator(rainfall, elninos, cutoff, roll_count):

    merg = volcano_rain_frame(rainfall, roll_count)
    ninas = pd.DataFrame()

    for i in elninos[cutoff]:
        ninas = pd.concat([ninas, merg[(merg['Decimal'] <= i[1]) & (merg['Decimal'] >= i[0])]])
        merg = merg[~((merg['Decimal'] <= i[1]) & (merg['Decimal'] >= i[0]))]

    return merg, ninas

def nino_rain_compare(oni, rainfall, rolling_number):

    new_rain = volcano_rain_frame(rainfall, rolling_number, None, None, True, False)
    average = new_rain['Precipitation'].mean()
    new_rain['roll'] = new_rain['roll'].apply(lambda x: (x/rolling_number) - average)

    new_nino = elnino_cleaner(oni)

    merged_df = pd.merge(new_rain, new_nino, left_on='Date', right_on='Center', how='inner')  

    return merged_df  

def nino_dict(oni, rainfall):
    start = min(rainfall['Date'])
    end = max(rainfall['Date'])
    strengths = {}
    anom_types = {'weak nina': -.5, 'moderate nina': -1, 'strong nina': -1.5, 'weak nino': .5, 'moderate nino': 1, 'strong nino': 1.5, 'very strong nino': 2}
    for i in anom_types:
        if i == 'weak nina' or i == 'weak nino':
            strengths.update(elnino_strengths(oni, anom_types[i], i, 5))
        else:
            strengths.update(elnino_strengths(oni, anom_types[i], i, 3)) 

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

# Creates histograms that break up eruption data based on quantile of rainfall.
def by_strength_nino(volcanos, eruptions, rainfall, color_count, roll_count, elninos=None, recur=False):

    # Plot distinction for one volcano versus multiple volcanos
    if len(volcanos) > 1:
        fig, axes = plt.subplots(len(volcanos), 1, figsize=(10, 20))
    else:
        fig, axes = plt.subplots(figsize=(10, 5))

    # Selects out color scheme
    plasma_colormap = cm.get_cmap('viridis', 256)
    if color_count > 1:
        color_spacing = 90 // (color_count-1)
        half_count = math.ceil(color_count / 2)
        upp_half = math.floor(color_count / 2)
        yellows = [plasma_colormap(255 - i*color_spacing)[:3] for i in range(half_count)]
        greens = [plasma_colormap(135 + i*color_spacing)[:3] for i in range(upp_half)]
        greens.reverse()
        colors = yellows + greens 
    else:
        colors = [plasma_colormap(210)]
    if color_count > 1:
        legend_handles = [mpatches.Patch(color=colors[i], label='Quantile ' + str(i+1)) for i in range(color_count)]
    else:
        legend_handles = [mpatches.Patch(color=colors[i], label='Rolling Precipitation') for i in range(color_count)]
    cmap = plt.cm.bwr
    selected_colors = cmap([253, 3])
    legend_handles += [mpatches.Patch(color=selected_colors[0], label='El Niño')]
    legend_handles += [mpatches.Patch(color=selected_colors[1], label='La Niña')]
    strengths = ['moderate nino', 'moderate nina']
    # legend_handles += [Line2D([0], [0], color='black', linestyle='dashed', dashes= (3,2), label='Volcanic event', linewidth= 1)]
    # legend_handles += [Line2D([0], [0], color=selected_colors[0], linestyle='dashed', dashes= (3,2), label='El Niño volcanic event', linewidth= 1)]
    # legend_handles += [Line2D([0], [0], color=selected_colors[1], linestyle='dashed', dashes= (3,2), label='La Niña volcanic event', linewidth= 1)]
    if recur == True:
        recurs = recurrences(eruptions, volcanos)

    # Creates a dictionary where for each volcano, we get an array of eruptions in each quantile.
    totals = {volcano:np.zeros(color_count) for volcano in volcanos}
    categories = ['Quantile ' + str(i+1) for i in range(color_count)]
    erupt_vals = {pick:[] for pick in totals}
    all_vals = {}
    count = 0
    for pick in totals:

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

        # Get volcano specific data and order dates by 'roll' amount
        dates = volc_rain.sort_values(by=['roll']).copy()
        dates.dropna()
        date_dec = np.array(dates['Decimal'])
        date_rain = np.array(dates['roll'])
        # Used in plotting to make y range similar to max bar height 


        all_vals[pick] = date_rain
        for k in erupt_dates:
            for i in range(len(date_dec)):
                if k == date_dec[i]:
                    erupt_vals[pick].append(date_rain[i])

        date_rain = np.log(date_rain + 1.25)
        y_max = np.max(date_rain)



        # Counts eruptions in each quantile
        if color_count > 1:
            bin_size = len(dates) // color_count
            for l in range(color_count):
                y = date_rain[l*(bin_size): (l+1)*bin_size]
                axes[count].bar(range(l*(bin_size), (l+1)*bin_size), y, color=colors[l])
                axes[count].set_title(str(pick))
                axes[count].set_xlabel('Day index when sorted by 90 day rain average')
                axes[count].set_ylabel('Rainfall (mm)')
        else:
            axes.bar(range(len(date_rain)), date_rain, color=colors[0], width=1)  

            # Plots El Nino
            # for i in range(len(date_rain)):
            #     for j in range(len(strengths)):
            #         for k in elninos[strengths[j]]:
            #             if date_dec[i] >= k[0] and date_dec[i] <= k[1]:
            #                 line_color = selected_colors[j]     
            #                 axes.plot(i, y_max + .125, color=line_color, marker='s', alpha=1.0, markersize=.75)
                 
            axes.set_title(str(pick))
            axes.set_xlabel('Day index when sorted by 90 day rain average')
            axes.set_ylabel('Rainfall (mm)')
                            
        
        # xticks = []
        # xlabels = []
        for i in range(len(date_dec)):
            if date_dec[i] in erupt_dates:
                # xticks.append(i)
                # xlabels.append(str(int((12 * (date_dec[i] % 1)) // 1)+1) + '/' + str(int(date_dec[i] // 1))[2:])
                date = date_dec[i]
                line_color = 'black'
                if elninos != None:
                    for j in range(len(strengths)):
                        for k in elninos[strengths[j]]:
                            if date >= k[0] and date <= k[1]:
                                line_color = selected_colors[j]
                axes[count].axvline(x=i, color=line_color, linestyle= 'dashed', dashes= (9,6), linewidth = 1)
                # axes[count].set_xticks(xticks)
                # axes[count].set_xticklabels(xlabels, rotation=45)

        if color_count > 1:
            axes[count].legend(handles=legend_handles, fontsize='small')
        else:
            axes.legend(handles=legend_handles, fontsize='small')
        count += 1

    plt.show()

    return all_vals, erupt_vals

# Find eruptions that don't occur in El Ninos
def non_nino_eruptions(eruptions, elninos):

    erupt_dates = np.array(eruptions['Decimal'])
    non_nino = []
    nino = []
    for i in erupt_dates:
        nino_erupt = False
        for j in elninos['strong nino']:
            if i >= j[0] and i <= j[1]:
                nino.append(i)
                nino_erupt = True
        if nino_erupt == False:
            non_nino.append(i)

    return non_nino, nino

# Strength finder
def elnino_strengths(oni, val, strength, cutoff):
    """ Determines the date ranges for El Nino or La Nina events of a specified strength.

    Typical classification of nino/nina events:
    anom_types = {'weak nina': -.5, 'moderate nina': -1, 'strong nina': -1.5, 'weak nino': .5, 'moderate nino': 1, 'strong nino': 1.5, 'very strong nino': 2}

    Args:
        oni: Oni dataframe from (https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt).
        val: Chosen cutoff sea surface temp for an El Nino event (eg. > 0.5 signifies all El Nino events).
        strength: Intensity of the event type considered (eg. for > 2, strength would be the string "very strong el nino")

    Return:
        period: A one key dictionary with key given by the strength, and value given by a list of pairs [start, end],
        where start is the beginning of an event, and end is the conclusion of an event.

    """

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
def elnino_cleaner(oni):
    """ Averages nearest 90 day sea surface temperatures to get a daily value.

    Args:
        oni: Oni dataframe from (https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt).
        rainfall: Dataframe that must have a 'Date' column. Used solely to get nino/nina events within the specified date range.

    Return:
        cleaned_oni: A new dataframe with dates and daily sea surface temperature values.

    """
    
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