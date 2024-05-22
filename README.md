# Precipitation Modelling

This repository provides a suite of functions that can be used to analyze and plot precipitation data in a variety of ways. All of the tools for such plotting/analysis can be found in the "Generalized Functions" folder.

This repository began as a collaboration space for a research group studying the effects of rainfall on volcanic activity in the Galapagos. Some of our analysis can be viewed in the "Notebooks" folder along with the original functions used in the "Notebook Functions" folder. Not all of the functions we used were easily adaptable to arbitrary lat/lon coordinates, but you can view them here in case they provide inspiration for new ways to view your data.

## Directions:

    1. Download the satellite IMERG data...(fill in details for how)

    2. If analyzing volcanic activity, download volcano data (fill in details for how)

    3. If analyzing El Ninos/La Ninas, download data (fill in details for how)

    4. Create paths to the folders that contain these data (eg. '/Volumes/T7Shield/Volcano/GetPrecipitation-main/data')

Plot_data notebooks

    - Plot_type='bar' -- Visualize rolling averages of rainfall using a bar plot, and simultaneously plot a cumulative sum of rainfall
    - Plot_type='seasonal' -- Visualize rolling averages of rainfall on a year to year basis
    - Optional Customizations
        - Rolling average number
        - Number of quantiles to breakup data into (color coded)
        - Can add an eruption dataframe which adds to 'bar' and 'seasonal' plots (option to make a histogram of eruptions by quantile)
        - Can add El Nino/La Nina dataframe which adds to 'bar' and 'seasonal' plots

Hindcast_regression notebooks

    -

Helper Scripts

    - Helper_functions.py

    - Plot_functions.py

    - El_nino_functions.py

How to download the precipitation data (TBD)

    - Specify latitude/longitude coordinates

How to make an eruption dataframe (TBD)