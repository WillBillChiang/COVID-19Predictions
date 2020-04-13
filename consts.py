from enum import Enum

class Consts(Enum):
    DATA_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
    DROP_COLS = ["UID","iso2","iso3","code3","FIPS","Admin2","Country_Region","Lat","Long_","Combined_Key"]

    
