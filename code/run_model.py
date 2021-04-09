import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import requests
import datetime
import torch
import torch.utils.data
import os
import importlib
import sys
import re
import pickle
from mpl_toolkits import mplot3d
from io import BytesIO
from math import log, exp, tan, atan, ceil
from PIL import Image

#from utils import dataset_utils
from utils import dataset_utils
from utils import createAISdata
#from utils import protobufDecoder
#from utils import plotting
from models import VRNN
from Config import config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")
#timestamp = datetime.datetime.fromtimestamp(update.t_epoch_sec).strftime('%d/%m/%Y %H:%M:%S')

#shiptypes = config.SHIPTYPE_CARGO + config.SHIPTYPE_FISHING + config.SHIPTYPE_PASSENGER +config.SHIPTYPE_TANKER + config.SHIPTYPE_SAILING + config.SHIPTYPE_PLEASURE
shiptypes = config.SHIPTYPE_CARGO + config.SHIPTYPE_TANKER
shipFileName = 'CargTank_test'
binedges = (config.LAT_EDGES, config.LON_EDGES, config.SOG_EDGES, config.COG_EDGES)
batch_size = 32

tracks = createAISdata.createAISdataset(
    {'ROI': (config.LAT_MIN, config.LAT_MAX, config.LON_MIN, config.LON_MAX), 
     'timeperiod': (config.T_MIN, config.T_MAX), 
     'maxspeed': config.SOG_MAX, 
     'navstatuses': config.MOV_NAV_STATUSES, 
     'shiptypes': shiptypes, 
     'binedges': binedges, 
     'minTrackLength': config.MIN_TRACKLENGTH,
     'maxTrackLength': config.MAX_TRACKLENGTH, 
     'minTrackUpdate': config.MIN_TRACKUPDATES, 
     'resampleFrequency': config.RESAMPLEFREQ
    },
    datasets_path = config.datasets_path,
    dataset_filename = config.datapath
)

with open(config.datasets_path + shipFileName + ".pkl", "wb") as f:
        pickle.dump(tracks, f)
        
##dataset_utils.makeDatasetSplits(shipFileName, '24hour_' + shipFileName) ##Probably need to change this function
