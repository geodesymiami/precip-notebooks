import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import math
import seaborn as sns
from matplotlib.lines import Line2D
from helper_functions import volcano_rain_frame, volcano_erupt_dates, date_to_decimal_year
import geopandas as gpd
from shapely.geometry import box
from shapely.geometry import box, Polygon, MultiPolygon

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

# Heat map of when eruptions occur
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
def by_strength(volcanos, eruptions, rainfall, color_count, roll_count):

    fig, axes = plt.subplots(len(volcanos), 1, figsize=(10, len(rainfall['Date'].unique())//400))

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
    erupt_vals = []
    count = 0
    for pick in totals:

        # Get volcano specific data and order dates by 'roll' amount
        volc_rain = volcano_rain_frame(rainfall, volcanos, pick, roll_count)
        dates = volc_rain.sort_values(by=['roll']).copy()
        dates.dropna()
        date_dec = np.array(dates['Decimal'])
        date_rain = np.array(dates['roll'])

        start = int(dates['Decimal'].min() // 1)
        end = int(dates['Decimal'].max() // 1)

        erupt_dates = volcano_erupt_dates(eruptions, pick, start, end)

        length = len(date_dec)

        # Counts eruptions in each quantile
        bin_size = len(dates) // color_count
        for l in range(color_count):
            y = date_rain[l*(bin_size): (l+1)*bin_size]
            axes[count].bar(range(l*(bin_size), (l+1)*bin_size), y, color=colors[l])
            axes[count].set_title(str(pick))

        for i in range(len(date_dec)):
            if date_dec[i] in erupt_dates:
                axes[count].axvline(x=i, color='black', linestyle= 'dashed', dashes= (9,6), linewidth = 1)
        count += 1
    plt.show()

    return erupt_vals

# Creates histograms that break up eruption data based on quantile of rainfall.
def eruption_counter(volcanos, eruptions, rainfall, color_count, roll_count, by_season=False):
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

    fig, axes = plt.subplots(len(volcanos) + 1, 1, figsize=(10, len(rainfall['Date'].unique())//400))

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
    erupt_vals = []

    for pick in totals:

        # Get volcano specific data and order dates by 'roll' amount
        volc_rain = volcano_rain_frame(rainfall, volcanos, pick, roll_count)
        dates = volc_rain.sort_values(by=['roll']).copy()
        dates.dropna()
        date_dec = np.array(dates['Decimal'])
        date_rain = np.array(dates['roll'])

        start = int(dates['Decimal'].min() // 1)
        end = int(dates['Decimal'].max() // 1)

        erupt_dates = volcano_erupt_dates(eruptions, pick, start, end)

        length = len(date_dec)
        for k in erupt_dates:
            for i in range(len(date_dec)):
                if k == date_dec[i]:
                    erupt_vals.append(date_rain[i])


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

# Plot all volcanos
def rain_plotter(plot_type, volcanos, rainfall, color_count, roll_count, eruptions, by_season=False, log_flag=True, elninos=None):
    """ Primary function for assembling rain data to plot. Calls 'annual_subplotter' and 'bar_subplotter' depending on what type of plot is wanted.

    Args:
        plot_type: Currently, can either be 'annual_subplotter' or 'bar_subplotter'.
        volcanos: A dictionary of sites (eg. sites_dict = {'Wolf': (-91.35, .05, 'Wolf'), 'Fernandina': (-91.45, -.45, 'Fernandina')}).
        rainfall: Satellite rain dataframe for volcanos in chosen region. 
        color_count: Number of quantiles to break rain data into.
        roll_count: Number of days to average rain over.
        eruptions: A dataframe with columns-- 'Volcano' and 'Start'. 'Start' is the beginning date of the eruption given as a string-- YYYY-MM-DD.
        by_season: Boolean for if quantiles should be made for every year separately, or across the entire date range at once.
        log_flag: Boolean for whether to use a log scale for the rain data.
        elninos: A dictionary of nino/nina dates where keys are the strength of events and a value is a list of lists of start/end dates for events.
        (elninos = {'weak nina': [], 'moderate nina': [], 'strong nina': [], 'weak nino': [], 'moderate nino': [], 'strong nino': [], 'very strong nino': []})

    Return:

    """

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

    # Sets plot dimensions based on plot_type
    if plot_type == 'bar':
        fig, axes = plt.subplots(len(volcanos), 1, figsize=(10, 18))
    elif plot_type == 'annual':
        fig, axes = plt.subplots(len(volcanos), 2, gridspec_kw={'width_ratios': [4, 1]}, figsize=(10, len(rainfall['Date'].unique())//300))

    # Creates a plot for each volcano
    for pick in volcanos:
        legend_handles = [mpatches.Patch(color=colors[i], label='Quantile ' + str(i+1)) for i in range(color_count)]
        volc_rain = volcano_rain_frame(rainfall, volcanos, pick, roll_count)
        dates = volc_rain.sort_values(by=['roll'])
        date_dec = np.array(dates['Decimal'])
        date_rain = np.array(dates['roll'])
        
        start = int(dates['Decimal'].min() // 1)
        end = int(dates['Decimal'].max() // 1)

        erupt_dates = volcano_erupt_dates(eruptions, pick, start, end)

        if plot_type == 'bar':
            bar_subplotter(dates, color_count, count, colors, axes, date_dec, erupt_dates, roll_count, volcanos, pick, start, legend_handles, end, date_rain, by_season, log_flag, elninos)
        
        elif plot_type == 'annual':
            annual_subplotter(volc_rain, erupt_dates, axes, count, date_dec, dates, color_count, colors, start, end, volcanos, pick, legend_handles, by_season, elninos)

        count += 1

    # Data plot
    plt.tight_layout()
    plt.show()

    return

def annual_subplotter(volc_rain, erupt_dates, axes, count, date_dec, dates, color_count, colors, start, end, volcanos, pick, legend_handles, by_season=False, elninos=None):
    """ Sub-function for plotting rain in horizontal bars: y-axis is year, and x-axis is month.

    Args:
        Necessary carry-overs from the rain_plotter function. (Need to clean up inputs a bit)

    Return:

    """
    # Plots eruptions
    volc_x = [((i) % 1) for i in erupt_dates]
    volc_y = [(i // 1) + .45 for i in erupt_dates]
    axes[count, 0].scatter(volc_x, volc_y, color='black', marker='v', s=30, label='Volcanic Events')
    eruption = axes[count, 0].scatter(volc_x, volc_y, color='black', marker='v', s=30, label='Volcanic Events')
    legend_handles += [eruption]

    # Plots rain by quantile, and if by_season is True, then also by year.
    for i in range(color_count):
        if by_season == True:
            for j in range(start, end + 1):
                dates_j = np.array([day for day in date_dec if (day // 1) == j])
                bin_size = len(dates_j) // color_count
                x = dates_j % 1
                y = dates_j // 1
                axes[count, 0].scatter(x[i*bin_size:(i+1)*bin_size], y[i*bin_size:(i+1)*bin_size], color=colors[i], marker='s', s =30)
        else:
            bin_size = len(dates) // color_count
            x = date_dec % 1
            y = date_dec // 1
            axes[count, 0].scatter(x[i*bin_size:(i+1)*bin_size], y[i*bin_size:(i+1)*bin_size], color=colors[i], marker='s', s =30)

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
                legend_handles += [mpatches.Patch(color=line_color, label='strong Niño')]
            elif j == 'very strong nino':
                line_color = 'black'
                legend_handles += [mpatches.Patch(color=line_color, label='very strong Niño')]
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
                    axes[count, 0].plot([x1, x2], [y1 - .17, y1 - .17], color=line_color, alpha=1.0, linewidth=3)
                else:
                    axes[count, 0].plot([x1, 1.0022], [y1 - .17, y1 - .17], color=line_color, alpha=1.0, linewidth=3)
                    axes[count, 0].plot([-.0022, x2], [y2 - .17, y2 - .17], color=line_color, alpha=1.0, linewidth=3)

    axes[count, 0].set_yticks([start + (2*k) for k in range(((end + 1 - start) // 2))], [str(start + (2*k)) for k in range(((end + 1 - start) // 2))])
    axes[count, 0].set_xticks([(1/12)*k for k in range(12)], ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'])
    axes[count, 0].set_xlabel("Month") 
    axes[count, 0].set_ylabel("Year") 
    axes[count, 0].set_title('Precipitation and volcanic events at ' + volcanos[pick][2]) 
    axes[count, 0].legend(handles=legend_handles, fontsize='small')

    # Creates a sideplot that shows total rainfall by year
    totals = []
    years = [i for i in range(start, end+1)]
    for i in years:
        totals.append(volc_rain['Precipitation'][volc_rain['Decimal'] // 1 == i].sum())
    axes[count, 1].set_title('Total (mm)') 
    axes[count, 1].barh(years, totals, height=.5, color='purple')
    axes[count, 1].set_yticks([start + (2*k) for k in range(((end + 1 - start) // 2))], [str(start + (2*k)) for k in range(((end + 1 - start) // 2))])

    return

def bar_subplotter(dates, color_count, count, colors, axes, date_dec, erupt_dates, roll_count, volcanos, pick, start, legend_handles, end, date_rain, by_season=False, log_flag=True, elninos=None):
    """ Sub-function for plotting rain in horizontal bars: y-axis is year, and x-axis is month.

    Args:
        Necessary carry-overs from the rain_plotter function. (Need to clean up inputs a bit)

    Return:

    """
    if log_flag == True:
        date_rain = np.log(date_rain + 1.25)

    y_max = np.max(date_rain)

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
                axes[count].bar(dates_j[l*(bin_size): (l+1)*bin_size], daterain_j[l*(bin_size): (l+1)*bin_size], color =colors[l], width = 0.01, alpha = 1)
        else:
            bin_size = len(dates) // color_count
            axes[count].bar(date_dec[l*(bin_size): (l+1)*bin_size], date_rain[l*(bin_size): (l+1)*bin_size], color =colors[l], width = 0.01, alpha = 1)

    # Plots cumulative rainfall in the same plot as the 90 day rain averages.
    legend_handles += [mpatches.Patch(color='gray', label='Cumulative precipitation')]
    ax2 = axes[count].twinx()
    ax2.bar(dates.Decimal, np.array(dates['cumsum']), color ='gray', width = 0.01, alpha = .05)
    ax2.set_ylabel("Cumulative precipitation (mm)", rotation=270, labelpad= 10)

    # Plots eruptions
    for line_x in erupt_dates:
        axes[count].axvline(x=line_x, color='black', linestyle= 'dashed', dashes= (9,6), linewidth = 1)
    
    legend_handles += [Line2D([0], [0], color='black', linestyle='dashed', dashes= (3,2), label='Volcanic event', linewidth= 1)]

    # Plots nino/nina events
    cmap = plt.cm.bwr
    selected_colors = cmap([128, 132, 203, 253, 127, 121, 3])
    if elninos != None:
        for j in elninos:
            if j == 'weak nino':
                line_color = selected_colors[0]
            elif j == 'moderate nino':
                line_color = selected_colors[1]
            elif j == 'strong nino':
                line_color = selected_colors[2]
            elif j == 'very strong nino':
                line_color = selected_colors[3]
                legend_handles += [mpatches.Patch(color=line_color, label='El Niño')]
            elif j == 'weak nina':
                line_color = selected_colors[4]
            elif j == 'moderate nina':
                line_color = selected_colors[5]
            elif j == 'strong nina':
                line_color = selected_colors[6]
                legend_handles += [mpatches.Patch(color=line_color, label='La Niña')]
            for i in range(len(elninos[j])):
                x1 = elninos[j][i][0]
                x2 = elninos[j][i][1]
                axes[count].plot([x1, x2], [y_max + .125, y_max + .125], color=line_color, alpha=1.0, linewidth=6) 

    axes[count].set_ylabel(str(roll_count) + " day rolling average precipitation (mm)")
    axes[count].set_xlabel("Year")
    ticks = int(((y_max * 4) // 1)) + 1
    axes[count].set_title(str(volcanos[pick][2]))
    axes[count].set_yticks(ticks=[.25*i for i in range(ticks)], labels=[.25*i for i in range(ticks)])
    axes[count].set_xticks(ticks=[start + (2*i) for i in range(((end - start) // 2) + 1)], labels=["'" + str(start + (2*i))[-2:] for i in range(((end - start) // 2) + 1)])

    axes[count].legend(handles=legend_handles, loc='upper left', fontsize='small')
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