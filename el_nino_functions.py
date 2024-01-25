import pandas as pd 
import numpy as np


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