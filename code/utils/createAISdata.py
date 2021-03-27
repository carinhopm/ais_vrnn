import os
import re
import datetime
import math
import progressbar
import pickle
import pandas as pd
import numpy as np

#from utils import protobufDecoder
from utils import dataset_utils
from Config import config
 
def FindMMSIs():   ##Your implementation of step 1
    
    mmsis = pd.DataFrame(columns=['MMSI','File']) #Allocate dataframe
        
    for file in progressbar.progressbar():  #For each JSON file
        data = ReadJSONfile #Read the JSON file

        #Make a DataFrame with columns timestamp, lat, lon, speed, course
        df = pd.DataFrame(columns=['timestamp','lat','lon','speed','course'])
        
        #Fill df with the information from the JSON path information (data['Path'])
        for msg in data['Path']:
            df = df.append({'timestamp': msg[0],'lat': msg[1],'lon': msg[2],'speed': msg[3],'course': msg[4]}, ignore_index=True)
        
        #For each row in df determine the applicable Navigation Status. and add this new column, navstatus, to df
        df['navstatus'] = 'other'
        statusIdx = 0
        for index, row in df.iterrows():
            if statusIdx+1 < len(data['statushist']) and row['timestamp'] >= data['statushist'][statusIdx+1]:
                statusIdx += 1
            row['navstatus'] = convertNavStatusToId(data['statushist'][statusIdx])
        
        #Filter for all params x = x[x[:,LAT]>=LAT_MIN] ect.
        lat_min, lat_max, lon_min, lon_max = ROI
        t_min, t_max = timePeriod
        df = df.loc[
            (df['timestamp']>=t_min) &
            (df['timestamp']<=t_max) &
            (df['lat']>=lat_min) &
            (df['lat']<=lat_max) &
            (df['lon']>=lon_min) &
            (df['lon']<=lon_max) &
            (df['speed']<=maxSpeed) &
            (df['navstatus'].isin(navTypes))
            ]

        #If rows left in dataframe and shiptype isin Shiptypes 
        if len(df.index) > 0 and data["shiptype"] is in shiptypes:
            new_row = {'MMSI': data["mmsi"],
                       'File': PathToTheJSONFile
                      } #Allocate new row for dataframe mmsis
            mmsis = mmsis.append(new_row, ignore_index=True) #Add the new row
    
    mmsis.sort_values(['MMSI'], inplace=True)
    
    return mmsis

def ReadAndJoinData(JSONfiles):  ##Your implementation of step 2.1
    
    dataframes = []
    
    for file in JSONfiles:
        data = ReadJSONfile(file) #Read the JSON file
        
        stype = data['shiptype']
        
        #Convert .path and .status to dataframe similar to before
        df =
        
        dataframes.append(df) #Append df to list

    df = pd.concat(dataframes) #Join all dataframes into 1
    df.sort_values(['timestamp'], inplace=True)
    
    return df, stype

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

def InterpolateTrackAndResample(df, frequency):
    
    df["timestamp"] = df["timestamp"].apply(datetime.datetime.fromtimestamp)
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

def createAISdataset(params, dataset_filename):
    
    maxUpdates = params['maxTrackLength']/params['resampleFrequency']
    minUpdates = params['minTrackLength']/params['resampleFrequency']
    fixedLength = params['maxTrackLength']==params['minTrackLength']
      
    print('Finding MMSIs in ROI')
    mmsis = FindMMSIs() #Step 1
    
    dataFileName = 'data/data_' + dataset_filename + '.pkl'
    with open(dataFileName,'wb') as dataFile:
        print('Processing MMSIs')
        indicies = []
        for mmsi in progressbar.progressbar(pd.unique(mmsis['MMSI'])): #Step 2
            tmp = mmsis.loc[mmsis['MMSI']==mmsi,:]

            data, shipType = ReadAndJoinData() #Step 2.1
            data = FilterDataFrame(data, params['ROI'], params['maxspeed'], params['timeperiod']) #Step 2.2
            data = FilterOutStationaryNavStatus(data) #Step 2.3
            data = SplitIntoTracks(data, 900) #Step 2.4
            data = RemoveShortTracks(data, params['minTrackLength'],minUpdates) #Step 2.5

            mmsiTracks = data.groupby('TrackNumber') #This actually carries out the splitting based on step 2.4
            for tracknum, track in mmsiTracks:         # For each track

                track = InterpolateTrackAndResample(track, params['resampleFrequency'])    #Step 2.6 

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
                        indicies.append(index)                                                  #Save the location of the pickled track
    
    track_indcies = {
        'indicies': indicies,
        'dataFileName': dataFileName,
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
    
    with open('data/datasetInfo_' + dataset_filename + '.pkl','wb') as file:
        pickle.dump(track_indcies, file)                                            #Save the location of all the tracks in another pickle
    
    return track_indcies