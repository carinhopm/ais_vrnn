import os
import re
import datetime
import math
import progressbar
import pandas as pd
import numpy as np

#from utils import protobufDecoder
from Config import config

def isInROI(ROI, position):
    
    lat_min, lat_max, lon_min, lon_max = ROI
    lat, lon = position
    
    return (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)

def isUnderMaxSpeed(maxspeed, speed):
    return speed <= maxspeed

def isInTimeperiod(timeperiod, timestamp):
    
    t_min, t_max = timeperiod
    
    return (timestamp >= t_min) & (timestamp <= t_max)

def isInShiptype(shipTypes, shipType):
    
    return shipType in shipTypes

def isInNavType(navTypes, navType):

    return navType in navTypes
    
def protobufToDict(update, shiptype):
    
    return {
        'timestamp': round(update.t_epoch_sec),
        'lat': update.lat_udeg/1000000,
        'lon': update.lon_udeg/1000000,
        'speed': update.speed_ms,
        'course': update.course_deg % 360, #Make modulo 360 to turn -1 into 359
        'navstatus': update.aisstatus,
        'shiptype': shiptype
        }

def filter_tracks(row, ROI, maxSpeed, timePeriod, shipTypes, navTypes):

    isInROI_ = isInROI(ROI, (row[1], row[2]))
    isUnderMaxSpeed_ = isUnderMaxSpeed(maxSpeed, row[3])
    isInTimePeriod_ = isInTimeperiod(timePeriod, row[0])
    isInShipTypes_ = isInShiptype(shipTypes, row[6])
    isInNavTypes_ = isInNavType(navTypes, row[5])
    
    return isInROI_ & isUnderMaxSpeed_ & isInTimePeriod_ & isInShipTypes_ & isInNavTypes_

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

def createAISdataset(params):
    
    maxUpdates = params['maxTrackLength']/params['resampleFrequency']

    #Implement functon to find all relavant mmsis
        
    tracks_final = {}
    print('Processing MMSIs')
    for mmsi in progressbar.progressbar(pd.unique(mmsis['MMSI'])):
        dict_key_num = 0

        #Implement function that reads all data for given mmsi. Would be best if it saved it to a dataframe
        
        
        data = FilterDataFrame(data, params['ROI'], params['maxspeed'], params['timeperiod'])
        data = FilterOutStationaryNavStatus(data)
        data = SplitIntoTracks(data, 900)
        data = RemoveShortTracks(data, params['minTrackLength'], params['minTrackUpdate'])

        mmsiTracks = data.groupby('TrackNumber')
        for tracknum, track in mmsiTracks:
            #Potentially write track to new protobuf

            track = InterpolateTrackAndResample(track, params['resampleFrequency'])
                        
            #Separate into equal pieces less than 24 hours
            #Split into Ceil(Duration/Maxlength) equal pieces
            num_tracks = math.ceil(len(track.index)/maxUpdates)
                
            #Encode and save each track
            for trackSegment in np.array_split(track, num_tracks):
                encodedTrack = FourHotEncode(trackSegment, params['binedges'])

                track_length = encodedTrack.shape[0]

                key = str(mmsi) + '_' + str(dict_key_num)
                dict_key_num += 1
                tracks_final[key] = (mmsi, shipType, track_length, trackSegment["timestamp"].to_list(), encodedTrack)
    
    return tracks_final