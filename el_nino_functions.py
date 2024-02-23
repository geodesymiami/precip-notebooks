import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from helper_functions import volcano_rain_frame, volcano_erupt_dates, date_to_decimal_year, recurrences


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
                    if oni_array[count][3] < val:
                        event = False
                        break
                if event == True:
                    start = oni_array[first][4]
                    while oni_array[count][3] >= val:
                        count += 1
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
                    if oni_array[count][3] > val:
                        event = False
                        break
                if event == True:
                    start = oni_array[first][4]
                    while oni_array[count][3] <= val:
                        count += 1
                    end = oni_array[count-1][4]
                    period[strength].append([start, end])
            else:
                count += 1
    
    return period



# El nino day value
def elnino_cleaner(oni, rainfall):
    """ Averages nearest 90 day sea surface temperatures to get a daily value.

    Args:
        oni: Oni dataframe from (https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt).
        rainfall: Dataframe that must have a 'Date' column. Used solely to get nino/nina events within the specified date range.

    Return:
        cleaned_oni: A new dataframe with dates and daily sea surface temperature values.

    """
    
    # Add a new column for the first day of each month in a string. (eg. for 'DJF' add a column for the 1st day of Dec, Jan, and Feb)
    date_converter = {'DJF': ('12', '03'), 'JFM': ('01', '04'), 'FMA': ('02', '05'), 'MAM': ('03', '06'), 'AMJ': ('04', '07'), 'MJJ': ('05', '08'), 'JJA': ('06', '09'), 'JAS': ('07', '10'), 'ASO': ('08', '11'), 'SON': ('09', '12'), 'OND': ('10', '01'), 'NDJ': ('11', '02')}
    
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
    orig['Center'] = oni.apply(convert_mid, axis=1)
    orig['End'] = orig.apply(convert_end, axis=1) 

    # Average sea surface temperatures from the rows where a date falls in the speficied date range.
    dates = list(np.sort(rainfall['Date'].unique()))
    indices = []

    for i in dates:

        index = orig['ANOM'][(orig['Start'] <= i) & (orig['End'] > i)].mean()
        indices.append(index)

    data = list(zip(dates, indices))
    cleaned_oni = pd.DataFrame(data, columns=['Date', 'ONI'])

    return cleaned_oni