import os
import datetime
import numpy as np

class config(object):
    
    datasets_path = 'C://Users//asm//OneDrive - Netcompany//University//Master Thesis//pickleFiles//'
    #datasets_path = '/zhome/e3/a/144459/workspace/ais_outlier_detection/data/CargTank_1911/'
    index_fileName = 'CargTank_1911_idxs.pkl'
    datapath = os.path.join(datasets_path,'CargTank_1911_idxs')
    
    T_OFFSET = int(datetime.datetime(2019, 1, 1, 0, 0, 0, 0).timestamp()) #2019-1-1 00:00:00.000
    T_MIN = int(datetime.datetime(2019, 1, 1, 0, 0, 0, 0).timestamp()) - T_OFFSET #2019-1-1 00:00:00.000
    T_MAX = int(datetime.datetime(2020, 12, 31, 23, 59, 59, 999).timestamp()) - T_OFFSET #2020-12-31 23:59:59.999
    
    LAT_MIN = int(54.5) # * 600000)
    LAT_MAX = 56.5 #* 600000
    LON_MIN = 13 #* 600000
    LON_MAX = 17 #* 600000
    SOG_MAX = int(15.5)  #* 10)
    
    LAT_MIN_PUBLIC = 45
    LAT_MAX_PUBLIC = 51
    LON_MIN_PUBLIC = -10
    LON_MAX_PUBLIC = 0
    SOG_MAX_PUBLIC = 15.5 #????
    
    LAT_RES = 0.1 #10
    LON_RES = 0.1 #10
    SOG_RES = 0.5
    COG_RES = 5
    
    LAT_EDGES = np.arange(LAT_MIN, LAT_MAX+(LAT_RES/10000), LAT_RES)
    LON_EDGES = np.arange(LON_MIN, LON_MAX+(LON_RES/10000), LON_RES)
    SOG_EDGES = np.arange(0, SOG_MAX+(SOG_RES/10000), SOG_RES)
    COG_EDGES = np.arange(0, 360+(COG_RES/10000), COG_RES)
    
    LAT_EDGES_PUBLIC = np.arange(LAT_MIN_PUBLIC, LAT_MAX_PUBLIC+(LAT_RES/10000), LAT_RES)
    LON_EDGES_PUBLIC = np.arange(LON_MIN_PUBLIC, LON_MAX_PUBLIC+(LON_RES/10000), LON_RES)
    SOG_EDGES_PUBLIC = np.arange(0, SOG_MAX_PUBLIC+(SOG_RES/10000), SOG_RES)
    COG_EDGES_PUBLIC = np.arange(0, 360+(COG_RES/10000), COG_RES)
    
    MIN_TRACKLENGTH = 14400 #Min track time length is 4 hours
    MIN_TRACKUPDATES = 24 #Min number of track time updates is 24
    
    MAX_TRACKLENGTH = 86400 #Max track length is 24 hours. 
    
    RESAMPLEFREQ = 600 #Resample frequancy in seconds
        
    #AIS INDEX MEANINGS
    STAT_NAV_STATUSES = [1, 5, 6]
    MOV_NAV_STATUSES = [0, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 95, 96, 97, 98, 99]
    
    SHIPTYPE_FISHING = [30]
    SHIPTYPE_TOWING = [31, 32, 52]
    SHIPTYPE_DREDGING = [33]
    SHIPTYPE_DIVING = [34]
    SHIPTYPE_MILITARY = [35]
    SHIPTYPE_SAILING = [36]
    SHIPTYPE_PLEASURE = [37]
    SHIPTYPE_PASSENGER = [60, 61, 62, 63, 64, 65, 66, 67, 68, 69]
    SHIPTYPE_CARGO = [70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
    SHIPTYPE_TANKER = [80, 81, 82, 83, 84, 85, 86, 87, 88, 89]

    #CONFIGURATION FOR MODEL
    LATENT_SIZE = 100

    #Factor the longitude and lattitude in multiplied by in the original data
    lat_lon_multiply_factor = 600000

    #Factor the SOG is multiplied by in the original data
    SOG_multiply_factor = 10