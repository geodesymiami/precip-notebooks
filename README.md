Precipitation Modelling

This package provides a suite of functions and notebooks for downloading, plotting, and analyzing IMERG precipitation data.

Main features:

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

    - Helper_scripts.py

    - Plot_functions.py

    - El_nino_functions.py

How to download the precipitation data (TBD)

    - Specify latitude/longitude coordinates

How to make an eruption dataframe (TBD)