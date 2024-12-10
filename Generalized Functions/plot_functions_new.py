import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import math
import seaborn as sns
from matplotlib.lines import Line2D
from helper_functions import volcano_rain_frame, volcano_erupt_dates, date_to_decimal_year, recurrences, create_map, color_scheme, quantile_name, extract_volcanoes, data_preload
import geopandas as gpd
from shapely.geometry import box
from shapely.geometry import box, Polygon, MultiPolygon
from matplotlib.ticker import ScalarFormatter
import os
import sys

elninos = {'weak nina': [[2000.4164, 2001.1233], [2005.8712, 2006.2], [2007.5342, 2008.4548], [2008.874, 2009.2], [2010.4521, 2011.3671], [2011.6192, 2012.2027], [2016.6219, 2016.9562], [2017.7863, 2018.2849], [2020.6219, 2021.2849], [2021.7041, 2023.0384]], 'moderate nina': [[2007.7041, 2008.2877], [2010.5342, 2011.1233], [2011.7863, 2011.9534], [2020.789, 2021.0384]], 'strong nina': [[2007.8712, 2008.1233], [2010.7041, 2010.9534]], 'weak nino': [[2002.4521, 2003.1233], [2004.6219, 2005.1233], [2006.7041, 2007.0384], [2009.6192, 2010.2], [2015.2, 2016.2877], [2018.7863, 2019.3671], [2023.4521, 2024.0384]], 'moderate nino': [[2002.7041, 2002.9534], [2009.7863, 2010.1233], [2015.4521, 2016.2027], [2023.5342, 2024.0384]], 'strong nino': [[2015.5342, 2016.2027]], 'very strong nino': [[2015.7041, 2016.1233]]} 

# Plot average rain for each day of the year
def average_daily(rainfall, roll_count=1):
    """ Line plot of the average rolling values by day of the year.

    Args:
        rainfall: Pandas dataframe with columns Date and Precipitation.
        roll_count: Number of days to average rain over
        
    Return:
    """
    
    rainfall = volcano_rain_frame(rain, roll_count)
    rainfall['MonthDay'] = rainfall['Decimal'].apply(lambda x: (x) % 1)
    days = np.unique(np.array(rainfall['MonthDay']))
    rain = np.zeros(len(days))
    for i in range(len(rain)):
        rain[i] = np.mean(np.array(rainfall['roll'][rainfall['MonthDay'] == days[i]]))
    
    plt.figure(figsize=(15, 6))
    plt.plot(days, rain)
    
    plt.xlabel('Month')
    plt.ylabel('Average Rain')
    plt.title('Average Rain by Day of Year at ' + 'tbd')
    plt.xticks([i/12 for i in range(12)], ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'])

    plt.show()
    return
   
# Creates histograms that break up eruption data based on quantile of rainfall.
def by_strength(rainfall, color_count=1, roll_count=1, eruptions=pd.DataFrame(), volcano=None, log=True):
    """ Plots the sorted rolling rain values and adds color based on quantile breakdown. Further, it plots the eruption data on top of this plot.

    Args:
        rainfall: Pandas dataframe with columns Date and Precipitation.
        color_count: Number of quantiles to break rain data into.
        roll_count: Number of days to average rain over.
        eruptions: Pandas dataframe with columns Volcano, Start, End, Max Explosivity.
        filename: A csv with columns-- 'Volcano' and 'Start'. 'Start' is the beginning date of the eruption given as a string-- YYYY-MM-DD.
        log: True if you want a log scale for the rain values

    Return:
    """

    volc_rain, erupt_dates, colors, quantile, legend_handles, start, end = data_preload(rainfall, roll_count, eruptions, color_count)

    # Order rain data by 'roll' amount
    dates = volc_rain.sort_values(by=['roll']).copy()
    dates.dropna()
    date_dec = np.array(dates['Decimal'])
    date_rain = np.array(dates['roll'])

    plt.figure(figsize=(10, 5))

    # Plots the ordered rain data
    if color_count > 1:
        bin_size = len(dates) // color_count
        for l in range(color_count):
            y = date_rain[l*(bin_size): (l+1)*bin_size]
            plt.bar(range(l*(bin_size), (l+1)*bin_size), y, color=colors[l], width=1.1)
    else:
        plt.bar(range(len(date_rain)), date_rain, color=colors[0], width=1)  

    # Plots the eruptions as dotted vertical lines
    if not eruptions.empty:
        legend_handles += [Line2D([0], [0], color='black', linestyle='dashed', dashes= (3,2), label='Volcanic event', linewidth= 1)]                    
        for i in range(len(date_dec)):
            if date_dec[i] in erupt_dates:
                line_color = 'black'
                plt.axvline(x=i, color=line_color, linestyle= 'dashed', dashes= (9,6), linewidth = 1)

    # Set plot properties
    plt.title('tbd')
    plt.xlabel('Days sorted by ' + str(roll_count) + ' day precipitation')
    plt.ylabel('Rainfall (mm)')
    if log == True:
        plt.yscale('log')
        plt.yticks([1, 10, 100, 1000])
    plt.legend(handles=legend_handles, fontsize='small')

    plt.show()

    return 

# Creates plot that break up eruption data based on quantile of rainfall.
def grid_search(volcanos, start_date, end_date, folder, volcano, quant_range=[20,30,40,50], roll_range=[30,60,90,120], grid='event count'):
    """ Creates a grid of values (based on the 'grid' input) for various choice of upper percentile and choice of rolling number.

    Args:
        volcanos: A dictionary of sites (eg. sites_dict = {'Wolf': (-91.35, .05, 'Wolf'), 'Fernandina': (-91.45, -.45, 'Fernandina')}).
        start_date: String YYYY-MM-DD for the first date of interest
        end_date: String YYYY-MM-DD for the last date of interest
        folder: Location where the rain data is stored
        filename: A csv with columns-- 'Volcano' and 'Start'. 'Start' is the beginning date of the eruption given as a string-- YYYY-MM-DD. 
        quant_range: List of upper percentiles to consider for the rain data.
        roll_range: List of days to average rain over.
        grid: Type of data catalog (event count, scale factor, p value)

    Return:

    """
    if volcano is not None:
        eruptions = extract_volcanoes(folder, volcano)
    else:
        print('An eruption file must be given.')
    
    data = np.zeros((len(quant_range), len(roll_range)))

    rains = {}
    for pick in volcanos:
        rains[pick] = create_map(volcanos[pick][1], volcanos[pick][0], [start_date, end_date], folder)

    for quant in range(len(quant_range)):
        for roll in range(len(roll_range)):

            # Creates a dictionary where for each volcano, we get an array of eruptions in each quantile.
            totals = {volcano:0 for volcano in volcanos}
            erupt_count = 0

            for pick in totals:

                # Get volcano specific data and order dates by 'roll' amount
                volc_rain = volcano_rain_frame(rains[pick], roll, volcanos[pick][0], volcanos[pick][1])
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

            if grid == 'event count':
                data[quant, roll] = int(all_volcs)

            elif grid == 'scale factor':
                data[quant, roll] = round((int(all_volcs) / erupt_count) / (quant_range[quant] / 100), 3)

            elif grid == 'p value':
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
            
            else:
                print('Not a valid grid type')
            
    xticklabels = roll_range
    yticklabels = quant_range
    
    # Create the heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(data, xticklabels=xticklabels, yticklabels=yticklabels, cmap='Blues', annot=True)
    plt.title('Upper period ' + grid)
    plt.xlabel('Roll count (days)')
    plt.ylabel('Upper rain period (%)')
    plt.show()

    return

# Creates histograms that break up eruption data based on quantile of rainfall.
def eruption_counter(rainfall, color_count=1, roll_count=1, eruptions=pd.DataFrame(), by_season=False):
    """ For each volcano, breaks up rain data by amount, and bins eruptions based on this. Generates histograms for each volcano
    separately, and also a histogram that puts all of the eruption data together.

    Args:
        rainfall: Pandas dataframe with columns Date and Precipitation.
        color_count: Number of quantiles to break rain data into.
        roll_count: Number of days to average rain over.
        eruptions: Pandas dataframe with columns Volcano, Start, End, Max Explosivity.
        by_season: T if quantiles should be made for every year separately.

    Return:

    """

    colors = color_scheme(color_count)
    quantile = quantile_name(color_count)

    # Get volcano specific data and order dates by 'roll' amount
    volc_rain = volcano_rain_frame(rainfall, roll_count)
    start = int(volc_rain['Decimal'].min() // 1)
    end = int((volc_rain['Decimal'].max() // 1)+1)

    erupt_dates = volcano_erupt_dates(eruptions, start, end)

    # Creates a dictionary where for each volcano, we get an array of eruptions in each quantile.
    totals =  np.zeros(color_count)
    categories = ['Quantile ' + str(i+1) for i in range(color_count)]

    plt.figure(figsize=(10, len(rainfall['Date'].unique())//1000))
        
    dates = volc_rain.sort_values(by=['roll']).copy()
    dates = dates.dropna()
    date_dec = np.array(dates['Decimal'])

    # Counts eruptions in each quantile
    for l in range(color_count):
        if by_season == True:
            for j in range(start, end + 1):
                dates_j = np.array([day for day in date_dec if (day // 1) == j])
                bin_size = len(dates_j) // color_count
                quantile = dates_j[l*(bin_size): (l+1)*bin_size]
                for k in erupt_dates:
                    if k in quantile:
                        totals[l] += 1
        else:
            bin_size = len(dates) // color_count
            quantile = date_dec[l*(bin_size): (l+1)*bin_size]
            for k in erupt_dates:
                    if k in quantile:
                        totals[l] += 1

    plt.bar(categories, totals, color=colors)
    plt.ylabel('Volcanic events')
    plt.title("Volcanic events by rain amount at " + 'location tbd')
    plt.yticks([i for i in range(int(np.max(totals)) + 1)])    
    plt.show()

    return

# Predicts rain based on averages
def rain_averager(rainfall, color_count=1, roll_count=1):
    """ Computes average rain by day of the year across a time interval, and plots it.

    Args:
        rainfall: Pandas dataframe with columns Date and Precipitation.
        color_count: Number of quantiles to break rain data into.
        roll_count: Number of days to average rain over.

    Return:
    """

    plt.figure(figsize=(8, 5))

    colors = color_scheme(color_count)
    quantile = quantile_name(color_count)

    if color_count > 1:
        legend_handles = [mpatches.Patch(color=colors[i], label=quantile + str(i+1)) for i in range(color_count)]

    # Creates a dataframe for rainfall at a single volcano, with new columns 'Decimal', 'roll', and 'cumsum' for 
    # decimal date, rolling average, and cumulative sum respectively.
    volc_init = volcano_rain_frame(rainfall, roll_count)

    volc_init['MonthDay'] = volc_init['Decimal'].apply(lambda x: (x) % 1)
    days = np.unique(np.array(volc_init['MonthDay']))
    rain = np.zeros(len(days))
    for i in range(len(rain)):
        rain[i] = np.mean(np.array(volc_init['roll'][volc_init['MonthDay'] == days[i]]))

    data = {'Precipitation': rain, 'Date': days}
    rain_frame = pd.DataFrame(data)
    sorted_frame = rain_frame.sort_values(by=['Precipitation'])

    days = np.array(sorted_frame['Date'])
    rain = np.array(sorted_frame['Precipitation'])

    # Create bar plot
    days = np.array(days)
    bin_size = len(days) // color_count
    for l in range(color_count):
        plt.bar(days[l*(bin_size): (l+1)*bin_size], rain[l*(bin_size): (l+1)*bin_size], color =colors[l], width = 0.01, alpha = 1)

    # Add labels and title
    features = {'title':'Average rain at ' + 'location tbd' + ' (' + str(int(volc_init['Decimal'].min() // 1)) + '-' + str(int(volc_init['Decimal'].max() // 1)) + ')', 
                'xtitle':'Day', 'ytitle': str(roll_count) + ' day precipitation (mm)', 'xticks': [(1/24 + i*(1/12)) for i in range(12)], 
                'yticks': None, 'xlabels': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 'ylabels': None}
    
    set_plot_properties(plt, features['title'], features['xtitle'], features['ytitle'], features['xticks'], 
                        features['yticks'], features['xlabels'], features['ylabels'])
    
    plt.legend(handles=legend_handles, loc='upper right', fontsize='small')

    # Show plot
    plt.show()
    return 

def annual_plotter(rainfall, color_count=1, roll_count=1, eruptions=pd.DataFrame(), ninos=None, by_season=False):
    """ Plots rain in horizontal bars: y-axis is year, and x-axis is month.

    Args:
        rainfall: Pandas dataframe with columns Date and Precipitation.
        color_count: Number of quantiles to break rain data into.
        roll_count: Number of days to average rain over.
        eruptions: Pandas dataframe with columns Volcano, Start, End, Max Explosivity.
        ninos: True if you want to include El Nino data
        by_season: True if quantiles should be made for every year separately.

    Return:
    """
    global elninos   

    volc_rain, erupt_dates, colors, quantile, legend_handles, start, end = data_preload(rainfall, roll_count, eruptions, color_count)

    fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 1]}, figsize=(10, ((len(rainfall['Date'].unique())//1200)))) # to get plot of combined data 1960-2020, take length of figsize and apply-> // 1.5)

    ax0 = axes[0]
    ax1 = axes[1]

    # Plots eruptions
    if len(erupt_dates) > 0:
        volc_x = [((i) % 1) for i in erupt_dates]
        volc_y = [(i // 1) + .5 for i in erupt_dates]
        ax0.scatter(volc_x, volc_y, color='black', marker='v', s=(219000 // (len(rainfall['Date'].unique()))), label='Volcanic Events')
        eruption = ax0.scatter(volc_x, volc_y, color='black', marker='v', s=(219000 // (len(rainfall['Date'].unique()))), label='Volcanic Events')
        legend_handles += [eruption]

    # Plots rain by quantile, and if by_season is True, then also by year.
    for i in range(color_count):
        if by_season == True:
            for j in range(start, end + 1):
                rain_by_year = volc_rain[volc_rain['Decimal'] // 1 == j].copy()
                rain_j = rain_by_year.sort_values(by=['roll'])
                dates_j = np.array([rain_j['Decimal']])
                bin_size = len(dates_j) // color_count
                x = dates_j % 1
                y = dates_j // 1
                ax0.scatter(x[i*bin_size:(i+1)*bin_size], y[i*bin_size:(i+1)*bin_size], color=colors[i], marker='s', s=(219000 // len(rainfall['Date'].unique())))
        else:
            dates = volc_rain.sort_values(by=['roll'])
            date_dec = np.array(dates['Decimal'])
            bin_size = len(dates) // color_count
            x = date_dec % 1
            y = date_dec // 1
            ax0.scatter(x[i*bin_size:(i+1)*bin_size], y[i*bin_size:(i+1)*bin_size], color=colors[i], marker='s', s=(219000 // len(rainfall['Date'].unique())))

    # Plots nino/nina events
    if ninos == True:
        colors = {'strong nino':['gray', 'Strong El Niño'], 'very strong nino':['black', 'Very strong Niño']}
        for j in elninos:
            if j == 'strong nino' or j == 'very strong nino':
                legend_handles += [mpatches.Patch(color=colors[j][0], label=colors[j][1])]
                for i in range(len(elninos[j])):
                    x1 = elninos[j][i][0] % 1
                    y1 = elninos[j][i][0] // 1
                    x2 = elninos[j][i][1] % 1
                    y2 = (elninos[j][i][1] // 1)
                    if y1 == y2:
                        ax0.plot([x1, x2], [y1 - .17, y1 - .17], color=colors[j][0], alpha=1.0, linewidth=(21900 // len(rainfall['Date'].unique())))
                    else:
                        ax0.plot([x1, 1.0022], [y1 - .17, y1 - .17], color=colors[j][0], alpha=1.0, linewidth=(21900 // len(rainfall['Date'].unique())))
                        ax0.plot([-.0022, x2], [y2 - .17, y2 - .17], color=colors[j][0], alpha=1.0, linewidth=(21900 // len(rainfall['Date'].unique())))

    # Creates a sideplot that shows total rainfall by year
    totals = []
    years = [i for i in range(start, end+1)]
    for i in years:
        totals.append(volc_rain['Precipitation'][volc_rain['Decimal'] // 1 == i].sum())
    ax1.barh(years, totals, height=.5, color='purple')

    # Set plot properties
    ax0.set_yticks([start + (2*k) for k in range(((end + 2 - start) // 2))], [str(start + (2*k)) for k in range(((end + 2 - start) // 2))])
    ax0.set_xticks([(1/24 + (1/12)*k) for k in range(12)], ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'])
    ax0.set_xlabel("Month") 
    ax0.set_ylabel("Year") 
    ax0.set_title('Tbd') 
    ax0.legend(handles=legend_handles, fontsize='small')
    ax1.set_title('Total (mm)') 
    ax1.set_yticks([start + (2*k) for k in range(((end + 1 - start) // 2))], [str(start + (2*k)) for k in range(((end + 1 - start) // 2))])

    plt.show()

    return 

def bar_plotter(rainfall, lat, lon, color_count=1, roll_count=1, eruptions=pd.DataFrame(), ninos=None, by_season=False, log_flag=True, centered=False, cumsum=True):
    """ Plots rolling rain temporally-- y-axis is rolling rain values, and x-axis is time.

    Args:
        rainfall: Pandas dataframe with columns Date and Precipitation.
        color_count: Number of quantiles to break rain data into.
        roll_count: Number of days to average rain over.
        eruptions: Pandas dataframe with columns Volcano, Start, End, Max Explosivity.
        ninos: True if you want to include El Nino data
        by_season: True if quantiles should be made for every year separately.
        log_flag: True to use a log scale for the rain data.
        centered: True to use a centered rolling sum
        cumsum: True to plot the cumulative rainfall in gray behind the front plot.

    Return:

    """

    global elninos

    if type(rainfall) == dict:
        rainfall = pd.DataFrame(rainfall.items(), columns=['Date', 'Precipitation'])

    # Convert array into single values
    rainfall['Precipitation'] = rainfall['Precipitation'].apply(lambda x: x[0][0][0])

    volc_rain, erupt_dates, colors, quantile, legend_handles, start, end = data_preload(rainfall, roll_count, eruptions, color_count) 
        
    # Applies a log scale to precipitation data if log_flag == True
    if log_flag == True:
        y_min = volc_rain['roll'].min()
        volc_rain['roll'] = volc_rain['roll'].apply(lambda x: math.log(x - y_min + 1))

    fig, plot = plt.subplots(figsize=(10, 4.5))

    # Plots 90 day rain averages, colored by quantile
    for l in range(color_count):
        if by_season == True:
            # Rain data is broken into quantiles, year by year, and plotted
            for j in range(start, end + 1):
                rain_by_year = volc_rain[volc_rain['Decimal'] // 1 == j].copy()
                rain_j = rain_by_year.sort_values(by=['roll'])
                dates_j = np.array([rain_j['Decimal']])
                daterain_j = np.array([rain_j['roll']])
                bin_size = len(dates_j) // color_count
                plot.bar(dates_j[l*(bin_size): (l+1)*bin_size], daterain_j[l*(bin_size): (l+1)*bin_size], color =colors[l], width = 0.01, alpha = 1)
        else:
            # Rain data is broken into quantiles, and plotted
            dates = volc_rain.sort_values(by=['roll'])
            date_dec = np.array(dates['Decimal'])
            date_rain = np.array(dates['roll'])
            bin_size = len(dates) // color_count
            plot.bar(date_dec[l*(bin_size): (l+1)*bin_size], date_rain[l*(bin_size): (l+1)*bin_size], color =colors[l], width = 0.01, alpha = 1)

    # Plots cumulative rainfall in the same plot as the 90 day rain averages.
    if cumsum == True:
        legend_handles += [mpatches.Patch(color='gray', label='Cumulative precipitation')]
        plot2 = plot.twinx()
        plot2.bar(dates.Decimal, np.array(dates['cumsum']), color ='gray', width = 0.01, alpha = .05)

    # Plots eruptions   
    if len(erupt_dates) > 0:
        for line_x in erupt_dates:
            plot.axvline(x=line_x, color='black', linestyle= 'dashed', dashes= (9,6), linewidth = 1)
        legend_handles += [Line2D([0], [0], color='black', linestyle='dashed', dashes= (3,2), label='Volcanic event', linewidth= 1)]

    # Used in Nino plotting to make y range similar to max bar height 
    y_max = np.max(volc_rain['roll'].max())
    ticks = int(((y_max * 1.5) // 1))

    # Plots nino/nina events
    if ninos == True:
        cmap = plt.cm.bwr
        colors = {'strong nino':[cmap(253), 'Strong El Niño'], 'strong nina':[cmap(3), 'Strong La Niña']}
        for j in elninos:
            if j == 'strong nino' or j == 'strong nina':
                legend_handles += [mpatches.Patch(color=colors[j][0], label=colors[j][1])]
                for i in range(len(elninos[j])):
                    x1 = elninos[j][i][0]
                    x2 = elninos[j][i][1]
                    plot.plot([x1, x2], [ticks - .125, ticks - .125], color=colors[j][0], alpha=1.0, linewidth=6) 

    # Set plot properties
    plot.set_ylabel(str(roll_count) + " day precipitation (mm)")
    plot.set_xlabel("Year")
    if eruptions.empty:
        plot.set_title(f'Latitude: {lat}, Longitude: {lon}')
    else:
        volcano = eruptions['Volcano'].unique()
        plot.set_title(f'{volcano[0]} - Latitude: {lat}, Longitude: {lon}')
    plot.set_yticks(ticks=[i for i in range(ticks)])
    plot.set_xticks(ticks=[start + (2*i) for i in range(((end - start) // 2) + 1)], labels=["'" + str(start + (2*i))[-2:] for i in range(((end - start) // 2) + 1)])
    plot2.set_ylabel("Cumulative precipitation (mm)", rotation=270, labelpad= 10)
    plt.legend(handles=legend_handles, loc='upper left', fontsize='small')
    plt.tight_layout()

    plt.show()

    return

