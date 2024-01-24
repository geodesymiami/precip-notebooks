import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import cm
import matplotlib.patches as mpatches
import statsmodels.api as sm
from lifelines import CoxTimeVaryingFitter
import math
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap

# Strength finder
def elnino_strengths(oni, val, type):

    date_converter = {'DJF': '01', 'JFM': '02', 'FMA': '03', 'MAM': '04', 'AMJ': '05', 'MJJ': '06', 'JJA': '07', 'JAS': '08', 'ASO': '09', 'SON': '10', 'OND': '11', 'NDJ': '12'}
    
    def convert_mid(row):
        return str(row['YR']) + '-' + date_converter[row['SEAS']] + '-15' 
    
    orig = oni.copy()
    orig['Center'] = oni.apply(convert_mid, axis=1)

    period = {type: []}
    oni_array = np.array(orig)
    count = 0
    while count < (oni_array.shape[0] - 5):
        if val > 0:
            if oni_array[count][3] >= val:
                first = count
                event = True
                for j in range(4):
                    count += 1
                    if oni_array[count][3] < val:
                        event = False
                        break
                if event == True:
                    start = oni_array[first][4]
                    while oni_array[count][3] >= val:
                        count += 1
                    end = oni_array[count-1][4]
                    period[type].append([start, end])
            else:
                count += 1
        elif val < 0:
            if oni_array[count][3] <= val:
                first = count
                event = True
                for j in range(4):
                    count += 1
                    if oni_array[count][3] > val:
                        event = False
                        break
                if event == True:
                    start = oni_array[first][4]
                    while oni_array[count][3] <= val:
                        count += 1
                    end = oni_array[count-1][4]
                    period[type].append([start, end])
            else:
                count += 1
    
    return period



# El nino data cleaner
def elnino_cleaner(oni, rainfall):
    
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

    dates = list(np.sort(rainfall['Date'].unique()))
    indices = []

    for i in dates:

        index = orig['ANOM'][(orig['Start'] <= i) & (orig['End'] > i)].mean()
        indices.append(index)

    data = list(zip(dates, indices))
    cleaned_oni = pd.DataFrame(data, columns=['Date', 'ONI'])

    return cleaned_oni

# Performs a cox regression for a chosen volcano
def cox_regressor(rainfall, eruptions, volcanos, roll_count, lower_cutoff, upper_cutoff, shift, lat_range, lon_range):

    list_volcs = list(volcanos.keys())
    cox = pd.DataFrame()

    for pick in list_volcs:

        volc_rain = volcano_rain_frame(rainfall, volcanos, pick, roll_count, lat_range, lon_range)
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


# Scatter plot of 90 day rain at volcano vs Ayora or Bellavista
def scatter_compare(rainfall, pick, compare_site, site_name, roll_count):

    site = data_cleaner(compare_site, roll_count) 

    compare_frame = rainfall.merge(site, on='Date', how='inner')

    plt.figure(figsize=(15,8))

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
def regressor(rainfall, volcanos, pick, compare_site, roll_count, print_summary):

    compare_frame = rainfall.merge(compare_site, on='Date', how='inner')

    X_constants = sm.add_constant(compare_frame['roll'])
    model_sm = sm.OLS(compare_frame['roll_two'], X_constants).fit()
    if print_summary == True:
        print(model_sm.summary())

    return model_sm

# Applies linear regression to generate a dataframe of predicted rainfall values
def rain_predictor(rainfall, volcanos, compare_site, roll_count, print_summary, lat_range, lon_range):

    pred_rain = pd.DataFrame()
    site = data_cleaner(compare_site, roll_count)

    for pick in volcanos:

        rain_frame = volcano_rain_frame(rainfall, volcanos, pick, roll_count, lat_range, lon_range)
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
def volcano_rain_frame(rainfall, volcanos, pick, roll_count, lat_range, lon_range):
    lat = volcanos[pick][1]
    lon = volcanos[pick][0]
    nearby_rain = rainfall[(abs(lon - rainfall['Longitude']) <= lon_range) & (abs(lat - rainfall['Latitude']) <= lat_range)].copy()
    dates = np.sort(nearby_rain['Date'].unique())
    averages = [[date, nearby_rain['Precipitation'][nearby_rain['Date'] == date].mean()] for date in dates]
    volc_rain = pd.DataFrame(averages, columns = ['Date', 'Precipitation'])
    volc_rain['Decimal'] = volc_rain.Date.apply(date_to_decimal_year)
    volc_rain = volc_rain.sort_values(by=['Decimal'])
    volc_rain['roll'] = volc_rain.Precipitation.rolling(roll_count).mean()
    volc_rain = volc_rain.dropna()
    volc_rain['cumsum'] = volc_rain.Precipitation.cumsum()
    #dates = volc_rain.sort_values(by=['roll'])
    return volc_rain

# Picks out all eruptions of a specific volcano beyond a certain date
def volcano_erupt_dates(eruptions, pick, start, end):
    volc_erupts = eruptions[eruptions['Volcano'] == pick].copy()
    volc_erupts['Decimal'] = volc_erupts.Start.apply(date_to_decimal_year)
    erupt_dates = np.array(volc_erupts['Decimal'][(volc_erupts['Decimal'] >= start) & (volc_erupts['Decimal'] <= end)])
    return erupt_dates


# Volcano longitude and latitudes are recorded in a dictionary. "Picks" is the list of volcanos whose eruptions will be considered.
def eruption_counter(volcanos, eruptions, rainfall, color_count, roll_count, lat_range, lon_range, by_season):
    fig, axes = plt.subplots(len(volcanos) + 1, 1, figsize=(10, len(rainfall['Date'].unique())//400))
    plasma_colormap = cm.get_cmap('viridis', 256)

    color_spacing = 90 // (color_count-1)
    half_count = math.ceil(color_count / 2)
    upp_half = math.floor(color_count / 2)
    yellows = [plasma_colormap(255 - i*color_spacing)[:3] for i in range(half_count)]
    reds = [plasma_colormap(135 + i*color_spacing)[:3] for i in range(upp_half)]
    reds.reverse()
    colors = yellows + reds
    totals = {volcano:np.zeros(color_count) for volcano in volcanos}
    categories = ['Quantile ' + str(i+1) for i in range(color_count)]

    for pick in totals:

        volc_rain = volcano_rain_frame(rainfall, volcanos, pick, roll_count, lat_range, lon_range)
        dates = volc_rain.sort_values(by=['roll']).copy()
        dates.dropna()
        date_dec = np.array(dates['Decimal'])

        start = int(dates['Decimal'].min() // 1)
        end = int(dates['Decimal'].max() // 1)

        erupt_dates = volcano_erupt_dates(eruptions, pick, start, end)

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

    all_volcs = np.sum(totals[pick] for pick in totals);
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

    return all_volcs

# Plot all volcanos
def rain_plotter(plot_type, volcanos, rainfall, color_count, roll_count, log, eruptions, elninos, lat_range, lon_range, by_season):

    count = 0
    plasma_colormap = cm.get_cmap('viridis', 256)
    color_spacing = 90 // (color_count-1)
    half_count = math.ceil(color_count / 2)
    upp_half = math.floor(color_count / 2)
    yellows = [plasma_colormap(255 - i*color_spacing)[:3] for i in range(half_count)]
    greens = [plasma_colormap(135 + i*color_spacing)[:3] for i in range(upp_half)]
    greens.reverse()
    colors = yellows + greens

    if plot_type == 'bar':
        fig, axes = plt.subplots(len(volcanos), 1, figsize=(10, 18))
    elif plot_type == 'annual':
        fig, axes = plt.subplots(len(volcanos), 2, gridspec_kw={'width_ratios': [4, 1]}, figsize=(10, len(rainfall['Date'].unique())//300))

    for pick in volcanos:
        legend_handles = [mpatches.Patch(color=colors[i], label='Quantile ' + str(i+1)) for i in range(color_count)]
        volc_rain = volcano_rain_frame(rainfall, volcanos, pick, roll_count, lat_range, lon_range)
        dates = volc_rain.sort_values(by=['roll'])
        date_dec = np.array(dates['Decimal'])
        date_rain = np.array(dates['roll'])
        
        start = int(dates['Decimal'].min() // 1)
        end = int(dates['Decimal'].max() // 1)

        erupt_dates = volcano_erupt_dates(eruptions, pick, start, end)

        if plot_type == 'bar':
            bar_subplotter(log, dates, color_count, count, colors, axes, date_dec, erupt_dates, elninos, roll_count, volcanos, pick, start, legend_handles, end, date_rain, by_season)
        
        elif plot_type == 'annual':
            annual_subplotter(volc_rain, erupt_dates, axes, count, date_dec, dates, color_count, elninos, colors, start, end, volcanos, pick, legend_handles, by_season)

        count += 1
    # Data plot
    plt.tight_layout()
    plt.show()

    return

def annual_subplotter(volc_rain, erupt_dates, axes, count, date_dec, dates, color_count, elninos, colors, start, end, volcanos, pick, legend_handles, by_season):
    volc_x = [((i) % 1) for i in erupt_dates]
    volc_y = [(i // 1) + .45 for i in erupt_dates]
    axes[count, 0].scatter(volc_x, volc_y, color='black', marker='v', s=30, label='Volcanic Events')
    eruption = axes[count, 0].scatter(volc_x, volc_y, color='black', marker='v', s=30, label='Volcanic Events')
    legend_handles += [eruption]
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

    if elninos != None:
        for j in elninos:
            if j == 'weak nino':
                line_color = 'gray'
                # legend_handles += [mpatches.Patch(color=line_color, label='weak nino')]
            elif j == 'moderate nino':
                line_color = 'gray'
                # legend_handles += [mpatches.Patch(color=line_color, label='moderate nino')]
            elif j == 'strong nino':
                line_color = 'gray'
                legend_handles += [mpatches.Patch(color=line_color, label='strong Niño')]
            elif j == 'very strong nino':
                line_color = 'black'
                legend_handles += [mpatches.Patch(color=line_color, label='very strong Niño')]
            elif j == 'weak nina':
                line_color = 'gray'
                # legend_handles += [mpatches.Patch(color=line_color, label='weak nina')]
            elif j == 'moderate nina':
                line_color = 'gray'
                # legend_handles += [mpatches.Patch(color=line_color, label='moderate nina')]
            elif j == 'strong nina':
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

    totals = []
    years = [i for i in range(start, end+1)]
    for i in years:
        totals.append(volc_rain['Precipitation'][volc_rain['Decimal'] // 1 == i].sum())
    axes[count, 1].set_title('Total (mm)') 
    axes[count, 1].barh(years, totals, height=.5, color='purple')
    axes[count, 1].set_yticks([start + (2*k) for k in range(((end + 1 - start) // 2))], [str(start + (2*k)) for k in range(((end + 1 - start) // 2))])

    return

def bar_subplotter(log, dates, color_count, count, colors, axes, date_dec, erupt_dates, elninos, roll_count, volcanos, pick, start, legend_handles, end, date_rain, by_season):
    legend_handles += [mpatches.Patch(color='gray', label='Cumulative precipitation')]
    if log == True:
        date_rain = np.log(date_rain + 1.25)

    y_max = np.max(date_rain)

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

    ax2 = axes[count].twinx()
    ax2.bar(dates.Decimal, np.array(dates['cumsum']), color ='gray', width = 0.01, alpha = .05)
    ax2.set_ylabel("Cumulative precipitation (mm)", rotation=270, labelpad= 10)

    for line_x in erupt_dates:
        axes[count].axvline(x=line_x, color='black', linestyle= 'dashed', dashes= (9,6), linewidth = 1)
    
    legend_handles += [Line2D([0], [0], color='black', linestyle='dashed', dashes= (3,2), label='Volcanic event', linewidth= 1)]
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
    
    axes[count].set_title(str(volcanos[pick][2]))
    axes[count].set_yticks(ticks=[.25*i for i in range(9)], labels=[.25*i for i in range(9)])
    axes[count].set_xticks(ticks=[start + (2*i) for i in range(((end - start) // 2) + 1)], labels=["'" + str(start + (2*i))[-2:] for i in range(((end - start) // 2) + 1)])

    axes[count].legend(handles=legend_handles, loc='upper left', fontsize='small')
    return