import os
import re
import datetime
import math
import progressbar
import pickle
import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split

#from utils import protobufDecoder
from utils import dataset_utils
from Config import config

import timeit
 
def FindMMSIs(directory, ROI, maxSpeed, timePeriod, navTypes, shiptypes):   ##Your implementation of step 1
    
    mmsis_list = []

    for file in progressbar.progressbar(os.listdir(directory)):  #For each JSON file
    
        fileName = os.path.join(directory,file)
        
        #Skip problematic JSONs
        if fileName[0] == '_':
            print("\nSkipping file ", fileName)
            continue
        elif '_TheRest_' in fileName:
            print("\nSkipping file ", fileName)
            continue
        
        print("\nProcessing filename: ", fileName)
        data = ReadJSONfile(fileName) #Read the JSON file
        
        #Fill df with the information from the JSON path information (data['Path'])
        path_list = []
        for i in range(len(data['path'])):
            msg = data['path'][i]
            #For each row determine if its navigation status is within 'navTypes'
            if 'statushist' in data.keys():
                statusHist = data['statushist']
                currentStatus = 'other'
                if msg[0] in statusHist.keys():
                    currentStatus = statusHist.get(msg[0]) # updates to new navigation status
                currentStatus = dataset_utils.convertNavStatusToId(currentStatus) # last known nav. status
            else:
                currentStatus = dataset_utils.convertNavStatusToId(data['lastStatus']) # last known nav. status
            if currentStatus in navTypes:
                path_list.append({'timestamp': int(msg[0]),'lat': int(msg[1]/config.lat_lon_multiply_factor),'lon': int(msg[2]/config.lat_lon_multiply_factor),'speed': int(msg[3]/config.SOG_multiply_factor),'course': int(msg[4]),'navstatus': currentStatus})
        df = pd.DataFrame(path_list)
        
        #Filter for all params x = x[x[:,LAT]>=LAT_MIN] ect.
        if len(df.index) > 0:
            lat_min, lat_max, lon_min, lon_max = ROI
            t_min, t_max = timePeriod
            df = df.loc[
                (df['timestamp']>=t_min) &
                (df['timestamp']<=t_max) &
                (df['lat']>=lat_min) &
                (df['lat']<=lat_max) &
                (df['lon']>=lon_min) &
                (df['lon']<=lon_max) &
                (df['speed']<=maxSpeed)
                ]

            #If rows left in dataframe and shiptype isin Shiptypes 
            if len(df.index) > 0 and int(dataset_utils.convertNameToShipType(data['shiptype'])) in shiptypes:
                new_row = {'MMSI': data["mmsi"],
                           'File': fileName
                          } #Allocate new row for dataframe mmsis
                mmsis_list.append(new_row) #Add the new row
        
    if len(mmsis_list) > 0:
        mmsis = pd.DataFrame(mmsis_list)
        mmsis.sort_values(['MMSI'], inplace=True)
    else:
        mmsis = pd.DataFrame(columns=['MMSI','File']) #Allocate empty dataframe
    
    return mmsis

def ReadAndJoinData(JSONfiles):  ##Your implementation of step 2.1
    
    dataframes = []
    
    for file in JSONfiles:
        data = ReadJSONfile(file) #Read the JSON file
        
        stype = int(dataset_utils.convertNameToShipType(data['shiptype']))
        
        #Convert .path and .status to dataframe similar to before
        path_list = []
        lat_lon_multiply_factor = config.lat_lon_multiply_factor
        SOG_multiply_factor = config.SOG_multiply_factor
        for msg in data['path']:
            if 'statushist' in data.keys():
                statusHist = data['statushist']
                currentStatus = 'other'
                if str(msg[0]) in statusHist.keys():
                    currentStatus = statusHist.get(str(msg[0])) # updates to new navigation status
                path_list.append({'timestamp': int(msg[0]),'lat': int(msg[1]/lat_lon_multiply_factor),'lon': int(msg[2]/lat_lon_multiply_factor),
                                'speed': int(msg[3]/SOG_multiply_factor),'course': int(msg[4]),
                                'navstatus': dataset_utils.convertNavStatusToId(currentStatus)})
            else:
                path_list.append({'timestamp': int(msg[0]),'lat': int(msg[1]/lat_lon_multiply_factor),'lon': int(msg[2]/lat_lon_multiply_factor),
                                'speed': int(msg[3]/SOG_multiply_factor),'course': int(msg[4]),
                                'navstatus': dataset_utils.convertNavStatusToId(data['lastStatus'])})
        df = pd.DataFrame(path_list)
        
        dataframes.append(df) #Append df to list

    df = pd.concat(dataframes) #Join all dataframes into 1
    df.sort_values(['timestamp'], inplace=True)
    
    return df, stype


def ReadJSONfile(file):
    
    with open(file) as f:
        
        ##
        data = json.load(f)
        
        return data[0]

def FilterDataFrame(df, ROI, maxSpeed, timePeriod):
    lat_min, lat_max, lon_min, lon_max = ROI
    t_min, t_max = timePeriod
    df = df.loc[
        (df['timestamp']>=t_min) &
        (df['timestamp']<=t_max) &
        (df['lat']>=lat_min) &
        (df['lat']<=lat_max) &
        (df['lon']>=lon_min) &
        (df['lon']<=lon_max) &
        (df['speed']<=maxSpeed)
        ]
    
    return df

def FilterOutStationaryNavStatus(df):
    
    df = df.loc[(~df['navstatus'].isin(config.STAT_NAV_STATUSES))]
    
    return df

def SplitIntoTracks(df, timediff):
    
    #Split when time difference greater than timediff seconds 
    group_ids = (df['timestamp'] > (df['timestamp'].shift() + timediff)).cumsum()
    df['TrackNumber'] = group_ids
    
    return df
    
def RemoveShortTracks(df, min_time, min_updates):

    tracks = df.groupby('TrackNumber')
    trackNums = []
    
    for tracknum, track in tracks:
        if (len(track) > min_updates) & ((track['timestamp'].iloc[-1] - track['timestamp'].iloc[0]) > min_time):
            trackNums.append(tracknum)
    
    df = df.loc[(df['TrackNumber'].isin(trackNums))]
    
    return df

def InterpolateTrackAndResample(df, frequency, offset):
    
    #Transform relative timestamps to real ones using the offset
    df["timestamp"] = df["timestamp"].apply(lambda x: datetime.datetime.fromtimestamp(x + offset))
    df = df.set_index('timestamp').resample(str(frequency) + 'S', origin='start').mean()
    df = df.reset_index(level=0, inplace=False)
    df = df.interpolate("linear")
    df["timestamp"] = df["timestamp"].apply(datetime.datetime.timestamp)
    
    return df

def createDenseVector(update, lat_edges, lon_edges, speed_edges, course_edges):
    lat_dim = len(lat_edges) - 1
    lon_dim = len(lon_edges) - 1
    sog_dim = len(speed_edges) - 1
    cog_dim = len(course_edges) - 1
    
    lat_idx = np.argmax(lat_edges > update['lat']) - 1
    lon_idx = np.argmax(lon_edges >  update['lon']) - 1
    sog_idx = np.argmax(speed_edges >  update['speed']) - 1
    cog_idx = np.argmax(course_edges >  update['course']) - 1

    data_dim = lat_dim + lon_dim + sog_dim + cog_dim 
    dense_vect = np.zeros(data_dim)

    dense_vect[lat_idx] = 1
    dense_vect[lat_dim + lon_idx] = 1
    dense_vect[lat_dim + lon_dim + sog_idx] = 1
    dense_vect[lat_dim + lon_dim + sog_dim + cog_idx] = 1

    return dense_vect

def FourHotEncode(track, edges):
    
    lat_edges, lon_edges, speed_edges, course_edges = edges
    
    EncodedTrack = track.apply(createDenseVector, axis = 1, args=(lat_edges, lon_edges, speed_edges, course_edges))
    EncodedTrack = np.array(EncodedTrack.to_list())
    
    return EncodedTrack

def dumpTrackToPickle(mmsi, shiptype, track, file):
    
    savedTrack = {
        'mmsi': mmsi,
        'shiptype': shiptype,
        'track_length': len(track.index),
        'lat': track["lat"].to_list(),
        'lon': track["lon"].to_list(),
        'speed': track["speed"].to_list(),
        'course': track["course"].to_list(),
        'timestamp': track["timestamp"].to_list()
    }
    
    index = file.tell()
    pickle.dump(savedTrack,file)
    
    return index

def createAISdataset(params, datasets_path, dataset_filename):
    
    start_total_time = timeit.default_timer()
    maxUpdates = params['maxTrackLength']/params['resampleFrequency']
    minUpdates = params['minTrackLength']/params['resampleFrequency']
    fixedLength = params['maxTrackLength']==params['minTrackLength']
    timeOffset = params['timeOffset']
      
    print('Finding MMSIs in ROI')
    mmsis = FindMMSIs(datasets_path, params['ROI'], params['maxspeed'], params['timeperiod'], params['navstatuses'], params['shiptypes']) #Step 1
    
    dataFileName = dataset_filename + '.pkl'
    with open(dataFileName,'wb') as dataFile:
        print('Processing MMSIs')
        indicies = []
        for mmsi in progressbar.progressbar(pd.unique(mmsis['MMSI'])): #Step 2
            tmp = mmsis.loc[mmsis['MMSI']==mmsi,:]

            data, shipType = ReadAndJoinData(tmp['File']) #Step 2.1
            data = FilterDataFrame(data, params['ROI'], params['maxspeed'], params['timeperiod']) #Step 2.2
            data = FilterOutStationaryNavStatus(data) #Step 2.3
            data = SplitIntoTracks(data, 900) #Step 2.4
            data = RemoveShortTracks(data, params['minTrackLength'],minUpdates) #Step 2.5

            mmsiTracks = data.groupby('TrackNumber') #This actually carries out the splitting based on step 2.4
            for tracknum, track in mmsiTracks:         # For each track

                track = InterpolateTrackAndResample(track, params['resampleFrequency'], timeOffset)    #Step 2.6 

                if fixedLength==True:
                    groups = track.groupby(np.arange(len(track.index))//maxUpdates) #Split ensure max length   ##Step 2.7
                    for _, trackSegment in groups:
                        if len(trackSegment.index)==maxUpdates:
                            #Save tracksegment
                            index = dumpTrackToPickle(mmsi, shipType, trackSegment, dataFile) #Save segment to pickle
                            indicies.append(index)                                            #Save the location of the pickled track
                else:
                    #Separate into equal pieces less than maxTrackLength
                    num_tracks = math.ceil(len(track.index)/maxUpdates) #Split into Ceil(Duration/Maxlength) equal pieces   ##Step 2.7
                    for trackSegment in np.array_split(track, num_tracks):
                        #Save each tracksegment
                        index = dumpTrackToPickle(mmsi, shipType, trackSegment, dataFile)      #Save segment to pickle
                        indicies.append(index)                                                 #Save the location of the pickled track
    
    #Extra step to split indicies into train and test sets
    trainIndicies, testIndicies = train_test_split(indicies, test_size=0.20, random_state=42)
    
    track_indcies = {
        'trainIndicies': trainIndicies,
        'testIndicies': testIndicies,
        'dataFileName': config.index_fileName,
        'ROI': params['ROI'], 
        'timeperiod': params['timeperiod'], 
        'maxspeed': params['maxspeed'], 
        'navstatuses': params['navstatuses'], 
        'shiptypes': params['shiptypes'], 
        'binedges': params['binedges'], 
        'minTrackLength': params['minTrackLength'],
        'maxTrackLength': params['maxTrackLength'], 
        'resampleFrequency': params['resampleFrequency']  
    }
    
    elapsed_total = timeit.default_timer() - start_total_time
    print('Total elapsed time used to generate the dataset:')
    print(elapsed_total)
    
    return track_indcies