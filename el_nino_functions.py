import pandas as pd 
import numpy as np

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