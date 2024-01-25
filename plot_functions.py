import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import math
from matplotlib.lines import Line2D
from helper_functions import volcano_rain_frame, volcano_erupt_dates

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


# Volcano longitude and latitudes are recorded in a dictionary. "Picks" is the list of volcanos whose eruptions will be considered.
def eruption_counter(volcanos, eruptions, rainfall, color_count, roll_count, by_season=False):
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

        volc_rain = volcano_rain_frame(rainfall, volcanos, pick, roll_count)
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
def rain_plotter(plot_type, volcanos, rainfall, color_count, roll_count, eruptions, by_season=False, log_flag=True, elninos=None):

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

def bar_subplotter(dates, color_count, count, colors, axes, date_dec, erupt_dates, roll_count, volcanos, pick, start, legend_handles, end, date_rain, by_season=False, log_flag=True, elninos=None):
    legend_handles += [mpatches.Patch(color='gray', label='Cumulative precipitation')]
    if log_flag == True:
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