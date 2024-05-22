import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import math
import seaborn as sns
from matplotlib.lines import Line2D
from helper_functions import volcano_rain_frame, volcano_erupt_dates, date_to_decimal_year, recurrences
from el_nino_functions import non_nino_eruptions
import geopandas as gpd
from shapely.geometry import box
from shapely.geometry import box, Polygon, MultiPolygon
from matplotlib.ticker import ScalarFormatter

def monthly_eruptions(volcanic_events, category_names, start, end):

    volcanic_events['Decimal'] = volcanic_events['Start'].apply(lambda x: int(x[5:7]))

    volcanic_events = volcanic_events[(volcanic_events['Start'] >= start) & (volcanic_events['Start'] <= end)]

    # Sample data
    categories = list(category_names.keys())
    values = [[((volcanic_events['Volcano'] == i) & (volcanic_events['Decimal'] == j)).sum() for i in categories] for j in range(1,13)] 

    # Initialize legend labels and handles
    legend_labels = []
    handles = []

    all = np.sum(values)

    # Define colors for each segment of the bars
    colors = [plt.cm.plasma(i) for i in np.linspace(0.2, 0.8, 4)]

    # Plot
    fig, ax = plt.subplots()

    # Iterate through each set of values
    for i in range(len(values)):
        # Initialize the bottom position of the bar segments
        bottom = 0
        # Iterate through each segment value in the bar
        for j in range(len(categories)):
            # Plot each segment of the bar
            bar = ax.bar(i, values[i][j], bottom=bottom, color=colors[j])
            # Update the bottom position for the next segment
            bottom += values[i][j]
            # Add legend labels and handles only if not already added
            if category_names[categories[j]] not in legend_labels:
                handles.append(bar)
                legend_labels.append(category_names[categories[j]])
                
    line = ax.axhline(y=(all/12), color='gray', linestyle='--')
    handles.append(line)
    legend_labels.append('Monthly average')

    ax.set_title('Eruptions by month' + ' (' + str(start[0:4]) + '-' + str(end[0:4]) + ')')
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of eruptions')

    # Set x-axis ticks and labels
    ax.set_xticks(range(len(values)), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_yticks(range(11))
    handles.reverse()
    legend_labels.reverse()
    # Add legend
    ax.legend(handles, legend_labels)

    # Show plot
    plt.show()

    return



# Heat map of when eruptions occur (Not currently used in repo)
def activity_by_day(eruptions, elninos):
    fig, axes = plt.subplots(3, 1, figsize=(12,15))
    eruptions['Decimal'] = eruptions.Start.apply(date_to_decimal_year)
    decimal_erupts = np.array(eruptions['Decimal'])
    x = [((i) % 1) for i in decimal_erupts]
    bin_edges = [i * 1/365 for i in range(365)]
    bin_indices = np.digitize(x, bin_edges)
    data1 = np.zeros((1, 365))
    for i in bin_indices:
        for j in range(20):
            data1[0][i-11+j] += 1

    decimal_erupts_non_nino, decimal_erupts_nino = non_nino_eruptions(eruptions, elninos)
    x = [((i) % 1) for i in decimal_erupts_non_nino]
    bin_edges = [i * 1/365 for i in range(365)]
    bin_indices = np.digitize(x, bin_edges)
    data2 = np.zeros((1, 365))
    for i in bin_indices:
        for j in range(20):
            data2[0][i-11+j] += 1

    x = [((i) % 1) for i in decimal_erupts_nino]
    bin_edges = [i * 1/365 for i in range(365)]
    bin_indices = np.digitize(x, bin_edges)
    data3 = np.zeros((1, 365))
    for i in bin_indices:
        for j in range(20):
            data3[0][i-11+j] += 1

    sns.heatmap(data1, ax=axes[0], cbar=True, cbar_kws={"label": "Eruption count"}, annot=False, fmt=".2f", cmap="YlGnBu")
    sns.heatmap(data2, ax=axes[1], cbar=True, cbar_kws={"label": "Eruption count"}, annot=False, fmt=".2f", cmap="YlGnBu", vmax=4)
    sns.heatmap(data3, ax=axes[2], cbar=True, cbar_kws={"label": "Eruption count"}, annot=False, fmt=".2f", cmap="YlGnBu", vmax=4)

    axes[0].set_title("All Eruptions")
    axes[1].set_title("Eruptions Not During El Niño Periods")
    axes[2].set_title("Eruptions During El Niño Periods")

    axes[0].set_xlabel("Month")
    axes[1].set_xlabel("Month")
    axes[2].set_xlabel("Month")

    axes[0].set_yticks([])
    axes[1].set_yticks([])
    axes[2].set_yticks([])

    axes[0].set_xticks([(1/12)*k*365 for k in range(12)], ['11', '12', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'])
    axes[1].set_xticks([(1/12)*k*365 for k in range(12)], ['11', '12', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'])
    axes[2].set_xticks([(1/12)*k*365 for k in range(12)], ['11', '12', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'])
    plt.tight_layout()
    plt.show()    
    return    

# Plot average rain for each day of the year
def average_daily(rainfall, pick):

    rainfall['MonthDay'] = rainfall['Decimal'].apply(lambda x: (x) % 1)
    days = np.unique(np.array(rainfall['MonthDay']))
    rain = np.zeros(len(days))
    for i in range(len(rain)):
        rain[i] = np.mean(np.array(rainfall['roll'][rainfall['MonthDay'] == days[i]]))
    
    plt.figure(figsize=(15, 6))
    plt.plot(days, rain)
    
    plt.xlabel('Month')
    plt.ylabel('Average Rain')
    plt.title('Average Rain by Day of Year at ' + pick)
    plt.xticks([i/12 for i in range(12)], ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'])

    plt.show()
    return

# Creates histograms that break up eruption data based on quantile of rainfall.
def by_strength_all(volcanos, eruptions, rainfall1, rainfall2, rainfall3, color_count, roll_count, log=True, elninos=None, recur=False):

    # Plot distinction for one volcano versus multiple volcanos
    fig, axes = plt.subplots(len(volcanos), 3, figsize=(25, 20))

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
    if color_count > 1:
        if color_count == 3:
            legend_handles = [mpatches.Patch(color=colors[0], label='Lower tertile'), mpatches.Patch(color=colors[1], label='Middle tertile'), mpatches.Patch(color=colors[2], label='Upper tertile')]

    cmap = plt.cm.bwr

    if len(eruptions) > 0:
        legend_handles += [Line2D([0], [0], color='black', linestyle='dashed', dashes= (3,2), label='Volcanic event', linewidth= 1)]

    # Creates a dictionary where for each volcano, we get an array of eruptions in each quantile.
    totals = {volcano:np.zeros(color_count) for volcano in volcanos}
    count = 0

    for pick in totals:

        count2 = 0

        for i in range(3):

            if count2 == 0:

                axis = axes[count,0]
                volc_rain = volcano_rain_frame(rainfall1, roll_count, volcanos[pick][0], volcanos[pick][1])

            elif count2 == 1:

                axis = axes[count,1]
                volc_rain = rainfall2
            
            elif count2 == 2:

                axis = axes[count,2]
                volc_rain = rainfall3

            
            start = int(volc_rain['Decimal'].min() // 1)
            end = int(volc_rain['Decimal'].max() // 1) +1

            if pick == 'Isabela (all)':
                all = True
            
            else:
                all = False

            erupt_dates = volcano_erupt_dates(eruptions[eruptions['Volcano'] == pick], start, end)

            # Get volcano specific data and order dates by 'roll' amount
            dates = volc_rain.sort_values(by=['roll']).copy()
            dates.dropna()
            date_dec = np.array(dates['Decimal'])
            date_rain = np.array(dates['roll'])
            # Used in plotting to make y range similar to max bar height 

            y_max = np.max(date_rain)



            # Counts eruptions in each quantile
            if color_count > 1:
                bin_size = len(dates) // color_count
                for l in range(color_count):
                    y = date_rain[l*(bin_size): (l+1)*bin_size]
                    axis.bar(range(l*(bin_size), (l+1)*bin_size), y, color=colors[l], width=1.1)
                if pick == 'Wolf':
                    if count2 == 0:
                        axis.set_title('2000-2023', fontsize=25)
                    elif count2 == 1:
                        axis.set_title('1964-2000', fontsize=25)
                    elif count2 == 2:
                        axis.set_title('1900-1964', fontsize=25)
                if pick == 'Isabela (all)':
                    axis.set_xlabel('Day index when sorted by ' + str(roll_count) + ' day precipitation', fontsize=15)
                if count2 == 2:
                    axis.yaxis.set_label_position('right')
                    axis.set_ylabel(str(roll_count) + ' day precipitation (mm)', fontsize=15, rotation=-90, ha='center', va='center', labelpad=7) # Move the ticks to the right
                if count2 == 0:
                    axis.set_ylabel(volcanos[pick][2], fontsize=25)
                if log == True:
                    axis.set_yscale('log')
                    axis.set_yticks([0.1, 1, 10, 100, 1000])
                    axis.set_ylim([0.1, 1000])
                            
            for i in range(len(date_dec)):
                if date_dec[i] in erupt_dates:
                    line_color = 'black'
                    axis.axvline(x=i, color=line_color, linestyle= 'dashed', dashes= (9,6), linewidth = 1)
            if count2 == 0 and count == 0:
                axis.legend(handles=legend_handles, fontsize='large')
            count2 += 1
        count += 1

    plt.show()

    return 

# Creates histograms that break up eruption data based on quantile of rainfall.
def by_strength(volcanos, eruptions, rainfall, color_count, roll_count, log=True, elninos=None, recur=False):

    rainfall['Precipitation'] = rainfall['Precipitation'].apply(lambda x: max(x, 0))

    # Plot distinction for one volcano versus multiple volcanos
    if len(volcanos) > 1:
        fig, axes = plt.subplots(len(volcanos), 1, figsize=(7.5, 20))
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
        if color_count == 3:
            legend_handles = [mpatches.Patch(color=colors[0], label='Lower tertile'), mpatches.Patch(color=colors[1], label='Middle tertile'), mpatches.Patch(color=colors[2], label='Upper tertile')]
        else:
            legend_handles = [mpatches.Patch(color=colors[i], label='Quantile ' + str(i+1)) for i in range(color_count)]
    else:
        legend_handles = [mpatches.Patch(color=colors[i], label=str(roll_count) + ' day precipitation') for i in range(color_count)]
    cmap = plt.cm.bwr
    selected_colors = cmap([253, 3])
    if elninos is not None:
        legend_handles += [mpatches.Patch(color=selected_colors[0], label='El Niño')]
        legend_handles += [mpatches.Patch(color=selected_colors[1], label='La Niña')]
    strengths = ['moderate nino', 'moderate nina']
    if len(eruptions) > 0:
        legend_handles += [Line2D([0], [0], color='black', linestyle='dashed', dashes= (3,2), label='Volcanic event', linewidth= 1)]
    # legend_handles += [Line2D([0], [0], color=selected_colors[0], linestyle='dashed', dashes= (3,2), label='El Niño volcanic event', linewidth= 1)]
    # legend_handles += [Line2D([0], [0], color=selected_colors[1], linestyle='dashed', dashes= (3,2), label='La Niña volcanic event', linewidth= 1)]
    # recurs = recurrences(eruptions, volcanos)

    # Creates a dictionary where for each volcano, we get an array of eruptions in each quantile.
    totals = {volcano:np.zeros(color_count) for volcano in volcanos}
    categories = ['Quantile ' + str(i+1) for i in range(color_count)]
    erupt_vals = {pick:[] for pick in totals}
    all_vals = {}
    count = 0
    for pick in totals:

        if len(volcanos) > 1:

            axis = axes[count]
        
        else:
            
            axis = axes

        volc_init = volcano_rain_frame(rainfall, roll_count, volcanos[pick][0], volcanos[pick][1])
        start = int(volc_init['Decimal'].min() // 1)
        end = int((volc_init['Decimal'].max() // 1)+1)

        erupt_dates = volcano_erupt_dates(eruptions[eruptions['Volcano'] == pick], start, end)
        if recur == True:
            volc_rain = volc_init.copy()
            for i in range(len(erupt_dates)):
                volc_rain = volc_rain[~((volc_init['Decimal'] > erupt_dates[i]) & (volc_init['Decimal'] < erupt_dates[i] + recur[pick]))].copy()
                
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
        y_max = np.max(date_rain)



        # Counts eruptions in each quantile
        if color_count > 1:
            bin_size = len(dates) // color_count
            for l in range(color_count):
                y = date_rain[l*(bin_size): (l+1)*bin_size]
                axis.bar(range(l*(bin_size), (l+1)*bin_size), y, color=colors[l], width=1.1)
                axis.set_title(volcanos[pick][2] + ' (' + str(start) + '-' + str(end-1) + ')')
                if count == len(volcanos)-1:
                    axis.set_xlabel('Day index when sorted by ' + str(roll_count) + ' day precipitation')
                axis.set_ylabel(str(roll_count) + ' day precipitation (mm)')
                if log == True:
                    axis.set_yscale('log')
        else:
            axis.bar(range(len(date_rain)), date_rain, color=colors[0], width=1)  

            # Plots El Nino
            # for i in range(len(date_rain)):
            #     for j in range(len(strengths)):
            #         for k in elninos[strengths[j]]:
            #             if date_dec[i] >= k[0] and date_dec[i] <= k[1]:
            #                 line_color = selected_colors[j]     
            #                 axes.plot(i, y_max + .125, color=line_color, marker='s', alpha=1.0, markersize=.75)
                 
            # axis.set_title(str(pick))
            axis.set_xlabel('Days sorted by ' + str(roll_count) + ' day precipitation')
            # axis.set_ylabel('Rainfall (mm)')
            if log == True:
                axis.set_yscale('log')
                axis.set_yticks([1, 10, 100, 1000])
                            
        
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
                axis.axvline(x=i, color=line_color, linestyle= 'dashed', dashes= (9,6), linewidth = 1)
                # axes[count].set_xticks(xticks)
                # axes[count].set_xticklabels(xlabels, rotation=45)

        axis.legend(handles=legend_handles, fontsize='small')
        count += 1

    plt.show()

    return all_vals, erupt_vals

# Creates plot that break up eruption data based on quantile of rainfall.
def cutoff_grid(volcanos, eruptions, rainfall, quant_range, roll_range, by_season=False, recur=False):
    """ For each volcano, breaks up rain data by amount, and bins eruptions based on this. Generates histograms for each volcano
    separately, and also a histogram that puts all of the eruption data together.

    Args:
        volcanos: A dictionary of sites (eg. sites_dict = {'Wolf': (-91.35, .05, 'Wolf'), 'Fernandina': (-91.45, -.45, 'Fernandina')}).
        eruptions: A dataframe with columns-- 'Volcano' and 'Start'. 'Start' is the beginning date of the eruption given as a string-- YYYY-MM-DD.
        rainfall: Satellite rain dataframe for volcanos in chosen region. 
        quant_range: List of quantiles to break rain data into.
        roll_range: List of days to average rain over.
        by_season: Boolean for if quantiles should be made for every year separately, or across the entire date range at once.

    Return:

    """
    data = np.zeros((len(quant_range), len(roll_range)))

    for quant in range(len(quant_range)):

        for roll in range(len(roll_range)):

            # Creates a dictionary where for each volcano, we get an array of eruptions in each quantile.
            totals = {volcano:0 for volcano in volcanos}
            erupt_count = 0


            for pick in totals:

                # Get volcano specific data and order dates by 'roll' amount
                volc_rain = volcano_rain_frame(rainfall, roll, volcanos[pick][0], volcanos[pick][1])
                start = int(volc_rain['Decimal'].min() // 1)
                end = int(volc_rain['Decimal'].max() // 1) + 1

                erupt_dates = volcano_erupt_dates(eruptions[eruptions['Volcano'] == pick], start, end)
                erupt_count += len(erupt_dates)

                    
                dates = volc_rain.sort_values(by=['roll']).copy()
                dates = dates.dropna()
                date_dec = np.array(dates['Decimal'])
                date_rain = np.array(dates['roll'])

                # Counts eruptions in each quantile
                bin_size = int((len(dates) * (quant_range[quant] / 100)) // 1)
                cutoff = date_rain[-bin_size]

            data[quant, roll] = cutoff

            # Computes the p-value


    xticklabels = roll_range
    yticklabels = quant_range
    
    # Create the heatmap
    plt.figure(figsize=(15, 6))
    sns.heatmap(data, xticklabels=xticklabels, yticklabels=yticklabels, cmap='Blues', annot=True)
    plt.title('Rain cutoff value')
    plt.xlabel('Roll count (days)')
    plt.ylabel('Upper rain period (%)')
    plt.show()

    return data

# Creates plot that break up eruption data based on quantile of rainfall.
def p_values(volcanos, eruptions, rainfall, quant_range, roll_range, by_season=False, recur=False):
    """ For each volcano, breaks up rain data by amount, and bins eruptions based on this. Generates histograms for each volcano
    separately, and also a histogram that puts all of the eruption data together.

    Args:
        volcanos: A dictionary of sites (eg. sites_dict = {'Wolf': (-91.35, .05, 'Wolf'), 'Fernandina': (-91.45, -.45, 'Fernandina')}).
        eruptions: A dataframe with columns-- 'Volcano' and 'Start'. 'Start' is the beginning date of the eruption given as a string-- YYYY-MM-DD.
        rainfall: Satellite rain dataframe for volcanos in chosen region. 
        quant_range: List of quantiles to break rain data into.
        roll_range: List of days to average rain over.
        by_season: Boolean for if quantiles should be made for every year separately, or across the entire date range at once.

    Return:

    """
    data = np.zeros((len(quant_range), len(roll_range)))

    for quant in range(len(quant_range)):

        for roll in range(len(roll_range)):

            # Creates a dictionary where for each volcano, we get an array of eruptions in each quantile.
            totals = {volcano:0 for volcano in volcanos}
            erupt_count = 0


            for pick in totals:

                # Get volcano specific data and order dates by 'roll' amount
                volc_rain = volcano_rain_frame(rainfall, roll, volcanos[pick][0], volcanos[pick][1])
                start = int(volc_rain['Decimal'].min() // 1)
                end = int(volc_rain['Decimal'].max() // 1) + 1

                erupt_dates = volcano_erupt_dates(eruptions[eruptions['Volcano'] == pick], start, end)
                erupt_count += len(erupt_dates)

                    
                dates = volc_rain.sort_values(by=['roll']).copy()
                dates = dates.dropna()
                date_dec = np.array(dates['Decimal'])

                # Counts eruptions in each quantile
                bin_size = int((len(dates) * (quant_range[quant] / 100)) // 1)
                quantile = date_dec[-bin_size:]
                for k in erupt_dates:
                    if k in quantile:
                        totals[pick] += 1

            percent = quant_range[quant]/100
            expected = erupt_count * (percent)
            all_volcs = sum([totals[pick] for pick in totals])
            deviation = abs(expected - all_volcs)


            upper = math.ceil(expected + deviation)
            lower = math.floor(expected - deviation)

            p = 0
            for i in range(upper, erupt_count+1):
                p += math.comb(erupt_count, i) * ((percent)**i) * ((1-(percent))**(erupt_count-i))
            for i in range(lower+1):
                p += math.comb(erupt_count, i) * ((percent)**i) * ((1-(percent))**(erupt_count-i))

            data[quant, roll] = p

            # Computes the p-value


    xticklabels = roll_range
    yticklabels = quant_range
    
    # Create the heatmap
    plt.figure(figsize=(15, 6))
    sns.heatmap(data, xticklabels=xticklabels, yticklabels=yticklabels, cmap='Blues_r', annot=True)
    plt.title('Binomial test p-values')
    plt.xlabel('Roll count (days)')
    plt.ylabel('Upper rain period (%)')
    plt.show()

    return data

# Creates plot that break up eruption data based on quantile of rainfall.
def grid_search(volcanos, eruptions, rainfall, quant_range, roll_range, by_season=False, recur=False):
    """ For each volcano, breaks up rain data by amount, and bins eruptions based on this. Generates histograms for each volcano
    separately, and also a histogram that puts all of the eruption data together.

    Args:
        volcanos: A dictionary of sites (eg. sites_dict = {'Wolf': (-91.35, .05, 'Wolf'), 'Fernandina': (-91.45, -.45, 'Fernandina')}).
        eruptions: A dataframe with columns-- 'Volcano' and 'Start'. 'Start' is the beginning date of the eruption given as a string-- YYYY-MM-DD.
        rainfall: Satellite rain dataframe for volcanos in chosen region. 
        quant_range: List of quantiles to break rain data into.
        roll_range: List of days to average rain over.
        by_season: Boolean for if quantiles should be made for every year separately, or across the entire date range at once.

    Return:

    """
    data = np.zeros((len(quant_range), len(roll_range)))
    data2 = np.zeros((len(quant_range), len(roll_range)))

    for quant in range(len(quant_range)):
        for roll in range(len(roll_range)):

            # Creates a dictionary where for each volcano, we get an array of eruptions in each quantile.
            totals = {volcano:0 for volcano in volcanos}
            erupt_count = 0


            for pick in totals:

                # Get volcano specific data and order dates by 'roll' amount
                volc_rain = volcano_rain_frame(rainfall, roll, volcanos[pick][0], volcanos[pick][1])
                start = int(volc_rain['Decimal'].min() // 1)
                end = int(volc_rain['Decimal'].max() // 1)+1

                erupt_dates = volcano_erupt_dates(eruptions[eruptions['Volcano'] == pick], start, end)
                erupt_count += len(erupt_dates)
                    
                dates = volc_rain.sort_values(by=['roll']).copy()
                dates = dates.dropna()
                date_dec = np.array(dates['Decimal'])

                # Counts eruptions in each quantile
                bin_size = int((len(dates) * (quant_range[quant] / 100)) // 1)
                quantile = date_dec[-bin_size:]
                for k in erupt_dates:
                    if k in quantile:
                        totals[pick] += 1

            all_volcs = sum([totals[pick] for pick in totals])
            data[quant, roll] = round((int(all_volcs) / erupt_count) / (quant_range[quant] / 100), 3)
            data2[quant, roll] = int(all_volcs)

    xticklabels = roll_range
    yticklabels = quant_range
    
    # Create the heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(data, xticklabels=xticklabels, yticklabels=yticklabels, cmap='Blues', annot=True)
    plt.title('Upper period scaling factor')
    plt.xlabel('Roll count (days)')
    plt.ylabel('Upper rain period (%)')
    plt.show()
    plt.figure(figsize=(12, 6))
    sns.heatmap(data2, xticklabels=xticklabels, yticklabels=yticklabels, cmap='Blues', annot=True)
    plt.title('Number of events in upper period')
    plt.xlabel('Roll count (days)')
    plt.ylabel('Upper rain period (%)')
    plt.show()

    return data

# Creates histograms that break up eruption data based on quantile of rainfall.
def eruption_counter(volcanos, eruptions, rainfall, color_count, roll_count, by_season=False, recur=False):
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

    fig, axes = plt.subplots(len(volcanos) + 1, 1, figsize=(10, (len(volcanos)+1) * len(rainfall['Date'].unique())//2000))

    # Selects out color scheme
    plasma_colormap = cm.get_cmap('viridis', 256)
    color_spacing = 90 // (color_count-1)
    half_count = math.ceil(color_count / 2)
    upp_half = math.floor(color_count / 2)
    yellows = [plasma_colormap(255 - i*color_spacing)[:3] for i in range(half_count)]
    reds = [plasma_colormap(135 + i*color_spacing)[:3] for i in range(upp_half)]
    reds.reverse()
    colors = yellows + reds

    # Creates a dictionary where for each volcano, we get an array of eruptions in each quantile.
    totals = {volcano:np.zeros(color_count) for volcano in volcanos}
    categories = ['Quantile ' + str(i+1) for i in range(color_count)]
    erupt_vals = {pick:[] for pick in totals}
    recurs = recurrences(eruptions, volcanos)


    for pick in totals:

        # Get volcano specific data and order dates by 'roll' amount
        volc_init = volcano_rain_frame(rainfall, roll_count, volcanos[pick][0], volcanos[pick][1])
        start = int(volc_init['Decimal'].min() // 1)
        end = int((volc_init['Decimal'].max() // 1)+1)

        erupt_dates = volcano_erupt_dates(eruptions[eruptions['Volcano'] == pick], start, end)
        print(start, end)
        print(erupt_dates)
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


        # Counts eruptions in each quantile
        for l in range(color_count):
            if by_season == True:
                for j in range(start, end + 1):
                    dates_j = np.array([day for day in date_dec if (day // 1) == j])
                    bin_size = len(dates_j) // color_count
                    quantile = dates_j[l*(bin_size): (l+1)*bin_size]
                    for k in erupt_dates:
                        if k in quantile:
                            totals[pick][l] += 1
            else:
                bin_size = len(dates) // color_count
                quantile = date_dec[l*(bin_size): (l+1)*bin_size]
                for k in erupt_dates:
                        if k in quantile:
                            totals[pick][l] += 1

    # Creates an array of eruptions tallied over all volcanos. Then makes a histogram for this array, and then for each volcano individually.
    all_volcs = np.sum(totals[pick] for pick in totals)
    y_set = int(np.max(all_volcs))
    axes[0].bar(categories, all_volcs, color=colors)
    axes[0].set_ylabel('Volcanic events')
    axes[0].set_title("Volcanic events by rain amount at all volcanos")
    axes[0].set_yticks([i for i in range(y_set + 1)])
    count = 1 
    for i in totals:
        axes[count].bar(categories, totals[i], color=colors)
        axes[count].set_ylabel('Volcanic events')
        axes[count].set_title("Volcanic events by rain amount at " + str(volcanos[i][2]))
        axes[count].set_yticks([i for i in range(y_set + 1)])
        count += 1       
    plt.show()

    return erupt_vals

# Predicts rain based on averages
def rain_averager(rainfall, volcanos, eruptions, color_count, roll_count):

    if len(volcanos) > 1:
        fig, axes = plt.subplots(math.ceil(len(volcanos)/2), 2, figsize=(16, 10))
    else:
        fig, axes = plt.subplots(figsize=(8, 5))
    
    # Selects out color scheme
    count = 0
    plasma_colormap = cm.get_cmap('viridis', 256)
    color_spacing = 90 // (color_count-1)
    half_count = math.ceil(color_count / 2)
    upp_half = math.floor(color_count / 2)
    yellows = [plasma_colormap(255 - i*color_spacing)[:3] for i in range(half_count)]
    greens = [plasma_colormap(135 + i*color_spacing)[:3] for i in range(upp_half)]
    greens.reverse()
    colors = yellows + greens 
    count=0

    erupt_counter = {pick:[] for pick in volcanos}

    for pick in volcanos:

        if len(volcanos) > 1:
            axis = axes[count // 2, count % 2]
        else:
            axis = axes

        if color_count == 3:
            legend_handles = [mpatches.Patch(color=colors[0], label='Lower tertile'), mpatches.Patch(color=colors[1], label='Middle tertile'), mpatches.Patch(color=colors[2], label='Upper tertile')]
        else:
            legend_handles = [mpatches.Patch(color=colors[i], label='Quantile ' + str(i+1)) for i in range(color_count)]

        # Creates a dataframe for rainfall at a single volcano, with new columns 'Decimal', 'roll', and 'cumsum' for 
        # decimal date, rolling average, and cumulative sum respectively.
        volc_init = volcano_rain_frame(rainfall, roll_count, volcanos[pick][0], volcanos[pick][1])

        start = 1900
        end = 1965

        # Creates a numpy array of decimal dates for eruptions between a fixed start and end date.
        erupt_dates = volcano_erupt_dates(eruptions[eruptions['Volcano'] == pick], start, end)
        erupt_no_year = erupt_dates % 1

        volc_init['MonthDay'] = volc_init['Decimal'].apply(lambda x: (x) % 1)
        days = np.unique(np.array(volc_init['MonthDay']))
        rain = np.zeros(len(days))
        for i in range(len(rain)):
            rain[i] = np.mean(np.array(volc_init['roll'][volc_init['MonthDay'] == days[i]]))

        data = {'Precipitation': rain, 'Date': days}
        rain_frame = pd.DataFrame(data)
        sorted_frame = rain_frame.sort_values(by=['Precipitation'])

        out_rain = pd.DataFrame()
        for i in range(start, end):
            year_rain = pd.DataFrame({'Decimal': [i+j for j in days], 'roll': rain})
            out_rain = pd.concat([out_rain, year_rain])


        days = np.array(sorted_frame['Date'])
        rain = np.array(sorted_frame['Precipitation'])

        # Create bar plot
        # plt.bar(days, rain)
        days = np.array(days)
        bin_size = len(days) // color_count
        for l in range(color_count):
            if l == color_count - 1:
                large_rain = days[l*(bin_size): (l+1)*bin_size]
                print(np.min(large_rain), np.max(large_rain))
            axis.bar(days[l*(bin_size): (l+1)*bin_size], rain[l*(bin_size): (l+1)*bin_size], color =colors[l], width = 0.01, alpha = 1)

        # Add labels and title
        axis.set_xlabel('Day')
        axis.set_ylabel(str(roll_count) + ' day precipitation (mm)')
        axis.set_xticks([(1/24 + i*(1/12)) for i in range(12)], ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        axis.set_title('Average rain at ' + str(volcanos[pick][2]) + ' (' + str(int(volc_init['Decimal'].min() // 1)) + '-' + str(int(volc_init['Decimal'].max() // 1)) + ')')
        axis.legend(handles=legend_handles, loc='upper right', fontsize='small')

        dates = rain_frame.sort_values(by=['Precipitation'])
        date_dec = np.array(dates['Date'])

        for i in range(color_count):
            bin_size = len(date_dec) // color_count
            quant = date_dec[i*bin_size:(i+1)*bin_size]
            for j in erupt_no_year:
                if j in quant:
                    erupt_counter[pick].append((i, j))

        count += 1   

    # Show plot
    plt.show()
    return out_rain


def annual_plotter(volcanos, rainfall, color_count, roll_count, eruptions, by_season=False, log_flag=True, elninos=None, recur=False):
    """ Sub-function for plotting rain in horizontal bars: y-axis is year, and x-axis is month.

    Args:
        volcanos: A dictionary of sites (eg. sites_dict = {'Wolf': (-91.35, .05, 'Wolf'), 'Negra, Sierra': (-91.15, -.85, 'Sierra Negra')}).
            -- The keys are string names for Volcano names based on how they are stored in the volcano data. The third entry in each tuple
            is how you'd like the volcano name to appear in the plot. 
        rainfall: Rain dataframe with columns: 'Date', 'Longitude', 'Latitude', 'Precipitation'
        color_count: Number of quantiles to break rain data into.
        roll_count: Number of days to average rain over.
        eruptions: A dataframe with columns-- 'Volcano' and 'Start'. 'Start' is the beginning date of the eruption given as a string-- YYYY-MM-DD.
        by_season: Boolean for if quantiles should be made for every year separately, or across the entire date range at once.
        log_flag: Boolean for whether to use a log scale for the rain data.
        elninos: A dictionary of nino/nina dates where keys are the strength of events and a value is a list of lists of start/end dates for events.
        (elninos = {'weak nina': [], 'moderate nina': [], 'strong nina': [], 'weak nino': [], 'moderate nino': [], 'strong nino': [], 'very strong nino': []})

    Return:
    """
    fig, axes = plt.subplots(len(volcanos), 2, gridspec_kw={'width_ratios': [4, 1]}, figsize=(10, ((len(rainfall['Date'].unique())//1200) * len(volcanos)))) # to get plot of combined data 1960-2020, take length of figsize and apply-> // 1.5)
    # Selects out color scheme
    count = 0
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

    if color_count == 2:
        quantile = 'Half '
    elif color_count == 3:
        quantile = 'tertile '
    elif color_count == 4:
        quantile = 'quartile '
    else:
        quantile = 'Quantile '

    # Creates a plot for each volcano
    for pick in volcanos:
        if len(volcanos) > 1:
            ax0 = axes[count,0]
            ax1 = axes[count,1]
        else:
            ax0 = axes[0]
            ax1 = axes[1]
        if color_count == 3:
            legend_handles = [mpatches.Patch(color=colors[0], label='Lower tertile'), mpatches.Patch(color=colors[1], label='Middle tertile'), mpatches.Patch(color=colors[2], label='Upper tertile')]
        else:
            legend_handles = [mpatches.Patch(color=colors[i], label='Quantile ' + str(i+1)) for i in range(color_count)]

        # Creates a dataframe for rainfall at a single volcano, with new columns 'Decimal', 'roll', and 'cumsum' for 
        # decimal date, rolling average, and cumulative sum respectively.
        volc_init = volcano_rain_frame(rainfall, roll_count, volcanos[pick][0], volcanos[pick][1])

        start = int(volc_init['Decimal'].min() // 1)
        end = int((volc_init['Decimal'].max() // 1) + 1)

        # Creates a numpy array of decimal dates for eruptions between a fixed start and end date.
        erupt_dates = volcano_erupt_dates(eruptions[eruptions['Volcano'] == pick], start, end)

        # If recur == True, it removes data from within a certain period after each eruption to account for recurrence times.
        if recur == True:
            recurs = recurrences(eruptions, volcanos)
            volc_rain = volc_init.copy()
            for i in range(len(erupt_dates)):
                volc_rain = volc_rain[~((volc_init['Decimal'] > erupt_dates[i]) & (volc_init['Decimal'] < erupt_dates[i] + recurs[pick]))].copy()

        else:
            volc_rain = volc_init.copy()
        dates = volc_rain.sort_values(by=['roll'])
        date_dec = np.array(dates['Decimal'])
        date_rain = np.array(dates['roll'])

        # Plots eruptions
        volc_x = [((i) % 1) for i in erupt_dates]
        volc_y = [(i // 1) + .5 for i in erupt_dates]
        ax0.scatter(volc_x, volc_y, color='black', marker='v', s=(219000 // (len(rainfall['Date'].unique()))), label='Volcanic Events')
        eruption = ax0.scatter(volc_x, volc_y, color='black', marker='v', s=(219000 // (len(rainfall['Date'].unique()))), label='Volcanic Events')
        legend_handles += [eruption]

        # Plots rain by quantile, and if by_season is True, then also by year.
        for i in range(color_count):
            if by_season == True:
                for j in range(start, end + 1):
                    dates_j = np.array([day for day in date_dec if (day // 1) == j])
                    bin_size = len(dates_j) // color_count
                    x = dates_j % 1
                    y = dates_j // 1
                    ax0.scatter(x[i*bin_size:(i+1)*bin_size], y[i*bin_size:(i+1)*bin_size], color=colors[i], marker='s', s=(219000 // len(rainfall['Date'].unique())))
            else:
                bin_size = len(dates) // color_count
                x = date_dec % 1
                y = date_dec // 1
                ax0.scatter(x[i*bin_size:(i+1)*bin_size], y[i*bin_size:(i+1)*bin_size], color=colors[i], marker='s', s=(219000 // len(rainfall['Date'].unique())))

        # Plots nino/nina events
        if elninos != None:
            for j in elninos:
                if j == 'weak nino':
                    continue
                    line_color = 'gray'
                    # legend_handles += [mpatches.Patch(color=line_color, label='weak nino')]
                elif j == 'moderate nino':
                    continue
                    line_color = 'gray'
                    # legend_handles += [mpatches.Patch(color=line_color, label='moderate nino')]
                elif j == 'strong nino':
                    line_color = 'gray'
                    legend_handles += [mpatches.Patch(color=line_color, label='Strong Niño')]
                elif j == 'very strong nino':
                    line_color = 'black'
                    legend_handles += [mpatches.Patch(color=line_color, label='Very strong Niño')]
                elif j == 'weak nina':
                    continue
                    line_color = 'gray'
                    # legend_handles += [mpatches.Patch(color=line_color, label='weak nina')]
                elif j == 'moderate nina':
                    continue
                    line_color = 'gray'
                    # legend_handles += [mpatches.Patch(color=line_color, label='moderate nina')]
                elif j == 'strong nina':
                    continue
                    line_color = 'gray'
                    # legend_handles += [mpatches.Patch(color=line_color, label='strong nina')]
                for i in range(len(elninos[j])):
                    x1 = elninos[j][i][0] % 1
                    y1 = elninos[j][i][0] // 1
                    x2 = elninos[j][i][1] % 1
                    y2 = (elninos[j][i][1] // 1)
                    if y1 == y2:
                        ax0.plot([x1, x2], [y1 - .17, y1 - .17], color=line_color, alpha=1.0, linewidth=(21900 // len(rainfall['Date'].unique())))
                    else:
                        ax0.plot([x1, 1.0022], [y1 - .17, y1 - .17], color=line_color, alpha=1.0, linewidth=(21900 // len(rainfall['Date'].unique())))
                        ax0.plot([-.0022, x2], [y2 - .17, y2 - .17], color=line_color, alpha=1.0, linewidth=(21900 // len(rainfall['Date'].unique())))

        ax0.set_yticks([start + (2*k) for k in range(((end + 2 - start) // 2))], [str(start + (2*k)) for k in range(((end + 2 - start) // 2))])
        ax0.set_xticks([(1/24 + (1/12)*k) for k in range(12)], ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'])
        ax0.set_xlabel("Month") 
        ax0.set_ylabel("Year") 
        ax0.set_title('Precipitation and volcanic events at ' + volcanos[pick][2]) 
        ax0.legend(handles=legend_handles, fontsize='small')

        # Creates a sideplot that shows total rainfall by year
        totals = []
        years = [i for i in range(start, end+1)]
        for i in years:
            totals.append(volc_rain['Precipitation'][volc_rain['Decimal'] // 1 == i].sum())
        ax1.set_title('Total (mm)') 
        ax1.barh(years, totals, height=.5, color='purple')
        ax1.set_yticks([start + (2*k) for k in range(((end + 1 - start) // 2))], [str(start + (2*k)) for k in range(((end + 1 - start) // 2))])
        count += 1
    # Data plot
    plt.tight_layout()
    plt.show()

    return

def bar_plotter(volcanos, rainfall, color_count, roll_count, eruptions, by_season=False, log_flag=True, elninos=None, recur=False, centered=False, cumsum=True):
    """ Sub-function for plotting rain in horizontal bars: y-axis is year, and x-axis is month.

    Args:
        volcanos: A dictionary of sites (eg. volcanos = {'Wolf': (-91.35, .05, 'Wolf'), 'Negra, Sierra': (-91.15, -.85, 'Sierra Negra')}).
            -- The keys are string names for Volcano names based on how they are stored in the eruptions dataframe (see below description of 'eruptions'). The third entry in each tuple
            is how you'd like the volcano name to appear in the plot. 
        rainfall: Rain dataframe with columns: 'Date', 'Longitude', 'Latitude', 'Precipitation'
        color_count: Number of quantiles to break rain data into.
        roll_count: Number of days to average rain over.
        eruptions: A dataframe with columns-- 'Volcano' and 'Start'. 'Start' is the beginning date of the eruption given as a string-- YYYY-MM-DD.
        by_season: Boolean for if quantiles should be made for every year separately, or across the entire date range at once.
        log_flag: Boolean for whether to use a log scale for the rain data.
        elninos: A dictionary of nino/nina date ranges, where keys are the strength of events and a value is a list of lists of start/end dates for events (as decimals).
        (elninos = eg. {'weak nina': [], 'moderate nina': [[2007.7041, 2008.2877], [2020.789, 2021.0384]], 'strong nina': [[2007.8712, 2008.1233], [2010.7041, 2010.9534]], 'weak nino': [], 'moderate nino': [], 'strong nino': [], 'very strong nino': []})

    Return:

    """
    # Plot distinction for one volcano versus multiple volcanos
    if len(volcanos) > 1:
        fig, axes = plt.subplots(len(volcanos), 1, figsize=(10, 4.5 * len(volcanos)))
    else:
        fig, axes = plt.subplots(figsize=(10, 4.5 * len(volcanos)))

    # Selects out color scheme
    count = 0
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
    if color_count == 2:
        quantile = 'half '
    elif color_count == 3:
        quantile = 'tertile '
    elif color_count == 4:
        quantile = 'quartile '
    else:
        quantile = 'quantile '

    # Creates a plot for each volcano
    for pick in volcanos:
        # Plot distinction for one volcano versus multiple volcanos
        if len(volcanos) > 1:
            axis = axes[count]
        else:
            axis = axes
        if color_count > 1:
            if color_count == 3:
                legend_handles = [mpatches.Patch(color=colors[0], label='Lower tertile'), mpatches.Patch(color=colors[1], label='Middle tertile'), mpatches.Patch(color=colors[2], label='Upper tertile')]
            else:
                legend_handles = [mpatches.Patch(color=colors[i], label='Quantile ' + str(i+1)) for i in range(color_count)]
        else:
            legend_handles = [mpatches.Patch(color=colors[i], label=str(roll_count) + ' day precipitation') for i in range(color_count)] 

        # Creates a dataframe for rainfall at a single volcano, with new columns 'Decimal', 'roll', and 'cumsum' for 
        # decimal date, rolling average, and cumulative sum respectively.
        volc_init = volcano_rain_frame(rainfall, roll_count, volcanos[pick][0], volcanos[pick][1])

        start = int(volc_init['Decimal'].min() // 1)
        end = int((volc_init['Decimal'].max() // 1)+1)

        # Creates a numpy array of decimal dates for eruptions between a fixed start and end date.
        erupt_dates = volcano_erupt_dates(eruptions[eruptions['Volcano'] == pick], start, end)

        # If recur == True, it removes data from within a certain period after each eruption to account for recurrence times.
        if recur == True:
            recurs = recurrences(eruptions, volcanos)
            volc_rain = volc_init.copy()
            for i in range(len(erupt_dates)):
                volc_rain = volc_rain[~((volc_init['Decimal'] > erupt_dates[i]) & (volc_init['Decimal'] < erupt_dates[i] + recurs[pick]))].copy()

        else:
            volc_rain = volc_init.copy()
        dates = volc_rain.sort_values(by=['roll'])
        date_dec = np.array(dates['Decimal'])
        date_rain = np.array(dates['roll'])
        y_min = np.min(date_rain)

        if log_flag == True:
            date_rain = np.log(date_rain - y_min + 1)

        # Used in plotting to make y range similar to max bar height 
        y_max = np.max(date_rain)
        ticks = int(((y_max * 1.5) // 1))

        # Plots cumulative rainfall in the same plot as the 90 day rain averages.
        if cumsum == True:
            legend_handles += [mpatches.Patch(color='gray', label='Cumulative precipitation')]
            ax2 = axis.twinx()
            ax2.bar(dates.Decimal, np.array(dates['cumsum']), color ='gray', width = 0.01, alpha = .05)
            ax2.set_ylabel("Cumulative precipitation (mm)", rotation=270, labelpad= 10)
            ax2.set_ylim(0, ax2.get_ylim()[1])

        # Plots 90 day rain averages, colored by quantile
        for l in range(color_count):
            if by_season == True:
                for j in range(start, end + 1):
                    dates_j = []
                    daterain_j = []
                    for k in range(len(date_dec)):
                        if date_dec[k] // 1 == j:
                            dates_j.append(date_dec[k])
                            daterain_j.append(date_rain[k])
                    bin_size = len(dates_j) // color_count
                    axis.bar(dates_j[l*(bin_size): (l+1)*bin_size], daterain_j[l*(bin_size): (l+1)*bin_size], color =colors[l], width = 0.01, alpha = 1)
            else:
                bin_size = len(dates) // color_count
                axis.bar(date_dec[l*(bin_size): (l+1)*bin_size], date_rain[l*(bin_size): (l+1)*bin_size], color =colors[l], width = 0.01, alpha = 1)

        # Plots eruptions
        for line_x in erupt_dates:
            axis.axvline(x=line_x, color='black', linestyle= 'dashed', dashes= (9,6), linewidth = 1)
        
        if len(erupt_dates) > 0:
            legend_handles += [Line2D([0], [0], color='black', linestyle='dashed', dashes= (3,2), label='Volcanic event', linewidth= 1)]

        # Plots nino/nina events
        cmap = plt.cm.bwr
        selected_colors = cmap([203, 253, 3])
        if elninos != None:
            for j in elninos:
                if j == 'strong nino' or j == 'very strong nino' or j == 'strong nina':
                    #if j == 'weak nino':
                    #    line_color = selected_colors[0]
                    #elif j == 'moderate nino':
                    #    line_color = selected_colors[1]
                    if j == 'strong nino':
                        line_color = selected_colors[0]
                    elif j == 'very strong nino':
                        line_color = selected_colors[1]
                        legend_handles += [mpatches.Patch(color=line_color, label='Strong El Niño')]
                    #elif j == 'weak nina':
                    #    line_color = selected_colors[4]
                    #elif j == 'moderate nina':
                    #    line_color = selected_colors[5]
                    elif j == 'strong nina':
                        line_color = selected_colors[2]
                        legend_handles += [mpatches.Patch(color=line_color, label='Strong La Niña')]
                    for i in range(len(elninos[j])):
                        x1 = elninos[j][i][0]
                        x2 = elninos[j][i][1]
                        axis.plot([x1, x2], [ticks - .125, ticks - .125], color=line_color, alpha=1.0, linewidth=6) 

        axis.set_ylabel(str(roll_count) + " day precipitation (mm)")
        axis.set_xlabel("Year")
        axis.set_title(str(volcanos[pick][2]))
        axis.set_yticks(ticks=[i for i in range(ticks)])
        axis.set_xticks(ticks=[start + (2*i) for i in range(((end - start) // 2) + 1)], labels=["'" + str(start + (2*i))[-2:] for i in range(((end - start) // 2) + 1)])

        axis.legend(handles=legend_handles, loc='upper left', fontsize='small')
        count += 1

    # Data plot
    plt.tight_layout()
    plt.show()
    return

# Used in spatial heat map below
def average_in_distance(arr, i, j, distance):
    rows, cols = arr.shape
    total = 0
    count = 0

    for x in range(max(0, i - distance), min(rows, i + distance + 1)):
        for y in range(max(0, j - distance), min(cols, j + distance + 1)):
            total += arr[x, y]
            count += 1

    return total / count

# Creates a heat map for average rain per day across a bounding box specified region
def spatial_heat_map(full_rain, season, long_min, long_max, lat_min, lat_max, granular=10):
    rain_in_region = full_rain[(full_rain['Longitude'] <= long_max) & (full_rain['Latitude'] <= lat_max) & (full_rain['Latitude'] >= lat_min) & (full_rain['Longitude'] >= long_min)]
    unique_trips = rain_in_region.groupby(['Longitude', 'Latitude', 'Decimal'])['Precipitation'].mean()
    to_plot = unique_trips.reset_index()
    longs = np.array(to_plot['Longitude'].drop_duplicates())
    lats = np.array(to_plot['Latitude'].drop_duplicates())
    lats = lats[::-1]
    data = np.zeros((len(lats), len(longs)))

    if season == 'wet':
        for i in range(len(lats)):
            for j in range(len(longs)):
                data[i][j] = to_plot['Precipitation'][(to_plot['Longitude'] == longs[j]) & (to_plot['Latitude'] == lats[i]) & ((to_plot['Decimal'] < 0.417) | (to_plot['Decimal'] >= 0.917))].sum()
    
    if season == 'dry':
        for i in range(len(lats)):
            for j in range(len(longs)):
                data[i][j] = to_plot['Precipitation'][(to_plot['Longitude'] == longs[j]) & (to_plot['Latitude'] == lats[i]) & ((to_plot['Decimal'] >= 0.417) & (to_plot['Decimal'] < 0.917))].sum()
    
    times_ten = np.repeat(data, granular, axis=0)
    times_ten = np.repeat(times_ten, granular, axis=1)

    rows, cols = times_ten.shape
    result_array = np.zeros_like(times_ten, dtype=float)

    distance = granular
    for i in range(rows):
        for j in range(cols):
            result_array[i, j] = average_in_distance(times_ten, i, j, distance)

    plt.figure(figsize=(9,7))
    sns.heatmap(result_array, annot=False, fmt=".2f", cmap="YlGnBu")
    contour = plt.contour(result_array, colors='k')
    plt.clabel(contour, inline=True, fontsize=8)

    plt.title("Average Cumulative Precipitation from December through May (mm)", fontsize='10')
    plt.xlabel("Longitude (degrees)", fontsize='10')
    plt.ylabel("Latitude (degrees)", fontsize='10')
    lat_range = int((((lat_max-lat_min)*10) // 2)) + 1
    long_range = int(((long_max - long_min)*10)) + 1
    plt.yticks([granular*2*i for i in range(lat_range)], [round((lat_max + .05) - (.1 * 2*i), 2) for i in range(lat_range)], rotation='horizontal')
    plt.xticks([granular*i for i in range(long_range)], [round((long_min-.05) + (.1 * i), 2) for i in range(long_range)], rotation=45)

    # points = [(1.5*granular, 5.5*granular), (3.5*granular, 2*granular), (3*granular, 11.25*granular), (5.75*granular, 10.25*granular), (5.75*granular,6.25*granular), (4.1*granular,4*granular)]
    # labels = ['F', 'W', 'CA', 'SN', 'A', 'D']

    # x, y = zip(*points)

    # sns.scatterplot(x=x, y=y, marker='o', color='red', s=100)

    # for i in range(len(labels)):
    #     plt.annotate(labels[i], (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize='20')

    shapefile_path = "/Users/jonathanquartin/Downloads/ECU_adm/ECU_adm0.shp"
    gdf = gpd.read_file(shapefile_path)

    # Create a bounding box geometry
    bbox = box(long_min, lat_min, long_max, lat_max)

    # Filter the GeoDataFrame based on the bounding box
    filtered_gdf = gdf[gdf.geometry.intersects(bbox)]

    for index, row in filtered_gdf.iterrows():
        geometry = row['geometry']
        coordinates = []
        if isinstance(geometry, MultiPolygon):
            for polygon in geometry.geoms:
                # Extract coordinates from each polygon
                x_coords, y_coords = polygon.exterior.xy
                coordinates.extend(list(zip(x_coords, y_coords)))
        elif isinstance(geometry, Polygon):
            # Extract coordinates from a single Polygon
            x_coords, y_coords = geometry.exterior.xy
            coordinates.extend(list(zip(x_coords, y_coords)))

        x = np.array([(i[0] - (long_min - .05)) * 100 for i in coordinates])
        y = np.array([(lat_max + .05 - i[1]) * 100 for i in coordinates])
        for i in range(len(x) - 1):
            if np.sqrt((x[i] - x[i+1])**2 + (y[i] - y[i+1])**2) < 10:
                plt.plot([x[i], x[i+1]], [y[i], y[i+1]], color='red', linestyle='-', linewidth=2)

    plt.show()
    return

### UNDER CONSTRUCTION ###
# Generates plots
# def custom_plot(ax, x, y, plot_type, title, xlabel, ylabel, **kwargs):
    
#     if plot_type == 'scatter':
#         ax.scatter(x, y, **kwargs)
#     elif plot_type == 'bar':
#         ax.bar(x, y, **kwargs)
#     elif plot_type == 'barh':
#         ax.barh(x, y, **kwargs)
#     else:
#         raise ValueError("Invalid plot type.")

#     ax.set_title(title)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_show()

#     return