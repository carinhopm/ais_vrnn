import pandas as pd
import numpy as np
import pickle
import os
import re
import datetime
import math
import torch

from utils import createAISdata

def convertShipTypeToName(shipType):
    
    choices = {
        '20': 'Wing In ground',
        '21': 'Wing In ground',
        '22': 'Wing In ground',
        '23': 'Wing In ground',
        '24': 'Wing In ground',
        '25': 'Wing In ground',
        '26': 'Wing In ground',
        '27': 'Wing In ground',
        '28': 'Wing In ground',
        '29': 'SAR Aircraft',
        '30': 'Fishing',
        '31': 'Tug',
        '32': 'Tug',
        '33': 'Dredger',
        '34': 'Dive Vessel',
        '35': 'Military',
        '36': 'Sailing',
        '37': 'Pleasure',
        '40': 'High Speed Vessel',
        '41': 'High Speed Vessel',
        '42': 'High Speed Vessel',
        '43': 'High Speed Vessel',
        '44': 'High Speed Vessel',
        '45': 'High Speed Vessel',
        '46': 'High Speed Vessel',
        '47': 'High Speed Vessel',
        '48': 'High Speed Vessel',
        '49': 'High Speed Vessel',
        '50': 'Pilot',
        '51': 'SAR Ship',
        '52': 'Tug',
        '53': 'Port Tender',
        '54': 'Anti-Pollution',
        '55': 'Law Enforcement',
        '56': 'Local Vessel',
        '57': 'Local Vessel',
        '58': 'Medical transfer',
        '59': 'Special Craft', #eg construction at windmills
        '60': 'Passenger',
        '61': 'Passenger',
        '62': 'Passenger',
        '63': 'Passenger',
        '64': 'Passenger',
        '65': 'Passenger',
        '66': 'Passenger',
        '67': 'Passenger',
        '68': 'Passenger',
        '69': 'Passenger',
        '70': 'Cargo',
        '71': 'Cargo',
        '72': 'Cargo',
        '73': 'Cargo',
        '74': 'Cargo',
        '75': 'Cargo',
        '76': 'Cargo',
        '77': 'Cargo',
        '78': 'Cargo',
        '79': 'Cargo',
        '80': 'Tanker',
        '81': 'Tanker',
        '82': 'Tanker',
        '83': 'Tanker',
        '84': 'Tanker',
        '85': 'Tanker',
        '86': 'Tanker',
        '87': 'Tanker',
        '88': 'Tanker',
        '89': 'Tanker',
        '90': 'Other',
        '91': 'Other',
        '92': 'Other',
        '93': 'Other',
        '94': 'Other',
        '95': 'Other',
        '96': 'Other',
        '97': 'Other',
        '98': 'Other',
        '99': 'Other'
    }
    
    return choices.get(shipType, 'Other')

def classNames():
    names = [
        'Cargo',
        'Tanker',
        'Fishing',
        'Passenger',
        'Sailing',
        'Pleasure',
        'High Speed Vessel',
        'Military',
        'Law Enforcement',
        'Pilot',
        'Tug',
        'Dredger',
        'Dive Vessel',
        'Port Tender',
        'Anti-Pollution',
        'Medical Transfer',
        'Special Craft',
        'Sar Ship',
        'Sar Aircraft',
        'Wing in Ground',
        'Other'
    ]
    
    return np.array(names), len(names)

class AISDataset(torch.utils.data.Dataset):
    def __init__(self, infoPath, train_mean = None):
        self.Infopath = infoPath
        
        self.classnames, _ = classNames()

        with open(self.Infopath, "rb") as f:
            self.params = pickle.load(f)
        
        if train_mean==None:
            self.indicies = self.params['trainIndicies']
        else:
            self.indicies = self.params['testIndicies']
        
        self.datapath = self.params['dataFileName']
        self.datasetN = len(self.indicies)
        
        lat_edges, lon_edges, speed_edges, course_edges = self.params['binedges']
        self.datadim = len(lat_edges) + len(lon_edges) + len(speed_edges) + len(course_edges) - 4
        
        if train_mean==None:
            self.mean = self.computeMean()
        else:
            self.mean = train_mean
            
        self.labels = self.getLabels()
        self.samples_pr_class = torch.bincount(self.labels)
        
    def __len__(self):
        return self.datasetN

    def __getitem__(self, idx):
            
        index = self.indicies[idx]
        
        with open(self.datapath, 'rb') as file:
            file.seek(index)
            track = pickle.load(file)
        
        tmpdf = pd.DataFrame(track)
        encodedTrack = createAISdata.FourHotEncode(tmpdf, self.params['binedges'])
        
        label = np.where(convertShipTypeToName(str(track['shiptype']))==self.classnames)[0][0]
        targets = torch.tensor(encodedTrack, dtype=torch.float) #seq_len X data_dim
        inputs = targets - self.mean
        
        return  torch.tensor(track['mmsi']), np.array(track['timestamp']), torch.tensor(label), torch.tensor(track['track_length'], dtype=torch.float), inputs, targets
    
    def computeMean(self):
        
        sum_all = np.zeros((self.datadim))
        total_updates = 0
        
        for index in self.indicies:
            with open(self.datapath,'rb') as file:
                file.seek(index)
                track = pickle.load(file)
                tmpdf = pd.DataFrame(track)
                encodedTrack = createAISdata.FourHotEncode(tmpdf, self.params['binedges'])
                sum_all += np.sum(encodedTrack,axis = 0) #Sum over all t
                total_updates += track['track_length']
        
        mean = sum_all/total_updates
        
        return torch.tensor(mean, dtype=torch.float)
    
    def getLabels(self):
        
        labels = []
        with torch.no_grad():
            for index in self.indicies:
                with open(self.datapath,'rb') as file:
                    file.seek(index)
                    track = pickle.load(file)
                    labels.append(np.where(convertShipTypeToName(str(track['shiptype']))==self.classnames)[0][0])
        
        return torch.tensor(labels)