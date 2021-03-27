import pandas as pd
import numpy as np
import pickle
import os
import re
import datetime
import math
import torch

def makeDatasetSplits(variableFilename, fixedFileName):
    #Split into test/train randomly
    def checkRow(row, mmsi, starttime, endtime):

        latestStart = max(row['starttime'], starttime)
        earliestEnd = min(row['endtime'], endtime)
        sameShip = row['mmsi'] == mmsi
        temporalOverlap = latestStart < earliestEnd

        return sameShip & temporalOverlap

    #Open 24hour and variable length datasets
    with open("data/all_" + variableFilename + ".pkl", "rb") as f:
            data = pickle.load(f)

    with open("data/all_" + fixedFileName + ".pkl", "rb") as f:
            data24hour = pickle.load(f)

    #Get mmsis and start + end time of 24 hour tracks
    rows = []
    keys = list(data24hour.keys())
    for key in keys:
        mmsi, _, _, timestamps, _ = data24hour[key]
        rows.append( {'key': key, 'mmsi': mmsi, 'starttime': timestamps[0], 'endtime': timestamps[-1]})
    df = pd.DataFrame(rows)

    # Sample 20% of the 24 hour tracks to a testset
    testSet = df.sample(frac=0.2, replace=False)

    #Retrive the keys of test and train sets
    testKeys = list(testSet['key'])
    trainKeys = [key for key in keys if key not in testKeys]

    #Construct train and test dicts
    train24hour = {key: data24hour[key] for key in trainKeys} 
    test24hour = {key: data24hour[key] for key in testKeys}

    #Save the dicts
    with open("data/train_" + fixedFileName + ".pkl", "wb") as f:
        pickle.dump(train24hour, f)
    with open("data/test_" + fixedFileName + ".pkl", "wb") as f:
        pickle.dump(test24hour, f)

    #Loop over all keys in variable length dataset
    keys = list(data.keys())
    testKeys = []
    for key in keys:
        mmsi, _, _, timestamps, _ = data[key]

        #Check for shared updates with 24 hour testset
        found = testSet.apply(checkRow, axis=1, args=(mmsi, timestamps[0], timestamps[-1])).any()

        #If they share updates with 24 hour testset
        if found:
            #Add to testset
            testKeys.append(key)

    trainKeys = [key for key in keys if key not in testKeys]

    #Construct train and test dicts
    train = {key: data[key] for key in trainKeys} 
    test = {key: data[key] for key in testKeys}

    #Save the dicts
    with open("data/train_" + variableFilename + ".pkl", "wb") as f:
        pickle.dump(train, f)
    with open("data/test_" + variableFilename + ".pkl", "wb") as f:
        pickle.dump(test, f)

def convertShipTypeToLabel(shipType):
    
    choices = {
        '20': 20, #Wing In ground 
        '21': 20, #Wing In ground
        '22': 20, #Wing In ground
        '23': 20, #Wing In ground
        '24': 20, #Wing In ground
        '25': 20, #Wing In ground
        '26': 20, #Wing In ground
        '27': 20, #Wing In ground
        '28': 20, #Wing In ground
        '29': 19, #SAR Aircraft
        '30': 3, #Fishing
        '31': 11, #Tug
        '32': 11, #Tug
        '33': 12, #Dredger
        '34': 13, #Dive Vessel
        '35': 8, #Military
        '36': 5, #Sailing
        '37': 6, #Pleasure
        '40': 7, #High Speed Vessel
        '41': 7, #High Speed Vessel
        '42': 7, #High Speed Vessel
        '43': 7, #High Speed Vessel
        '44': 7, #High Speed Vessel
        '45': 7, #High Speed Vessel
        '46': 7, #High Speed Vessel
        '47': 7, #High Speed Vessel
        '48': 7, #High Speed Vessel
        '49': 7, #High Speed Vessel
        '50': 10, #Pilot
        '51': 18, #SAR Ship
        '52': 11, #Tug
        '53': 14, #Port Tender
        '54': 15, #Anti-Pollution
        '55': 9, #Law Enforcement
        '56': 21, #Local Vessel
        '57': 21, #Local Vessel
        '58': 16, #Medical transfer
        '59': 17, #Special Craft. eg construction at windmills
        '60': 4, #Passenger
        '61': 4, #Passenger
        '62': 4, #Passenger
        '63': 4, #Passenger
        '64': 4, #Passenger
        '65': 4, #Passenger
        '66': 4, #Passenger
        '67': 4, #Passenger
        '68': 4, #Passenger
        '69': 4, #Passenger
        '70': 1, #Cargo
        '71': 1, #Cargo
        '72': 1, #Cargo
        '73': 1, #Cargo
        '74': 1, #Cargo
        '75': 1, #Cargo
        '76': 1, #Cargo
        '77': 1, #Cargo
        '78': 1, #Cargo
        '79': 1, #Cargo
        '80': 2, #Tanker
        '81': 2, #Tanker
        '82': 2, #Tanker
        '83': 2, #Tanker
        '84': 2, #Tanker
        '85': 2, #Tanker
        '86': 2, #Tanker
        '87': 2, #Tanker
        '88': 2, #Tanker
        '89': 2, #Tanker
        '90': 21, #Other
        '91': 21, #Other
        '92': 21, #Other
        '93': 21, #Other
        '94': 21, #Other
        '95': 21, #Other
        '96': 21, #Other
        '97': 21, #Other
        '98': 21, #Other
        '99': 21, #Other
    }
    
    return choices.get(shipType, 0)
        
class AISDataset(torch.utils.data.Dataset):
    def __init__(self, path, train_mean = None):
        self.path = path

        with open(self.path, "rb") as f:
            self.dataset = pickle.load(f)
        
        self.datasetKeys = list(self.dataset.keys())
        self.datasetN = len(self.datasetKeys)
        
        mmsi, shipType, sequenceLength, timestamps, inputs = self.dataset[self.datasetKeys[0]]
        self.datadim = inputs.shape[1]
        
        if train_mean==None:
            self.mean = self.computeMean()
        else:
            self.mean = train_mean
        
    def __len__(self):
        return self.datasetN

    def __getitem__(self, idx):
            
        key = self.datasetKeys[idx]
        
        mmsi, shipType, sequenceLength, timestamps, targets = self.dataset[key]
        
        label = convertShipTypeToLabel(shipType)
        targets = torch.tensor(targets, dtype=torch.float) #seq_len X data_dim
        inputs = targets - self.mean
        
        return  torch.tensor(mmsi),  torch.tensor(label),  torch.tensor(sequenceLength, dtype=torch.float), inputs, targets
    
    def computeMean(self):
        
        sum_all = np.zeros((self.datadim))
        total_updates = 0
        
        for i in range(self.datasetN):
            key = self.datasetKeys[i]
            mmsi, shipType, sequenceLength, timestamps, inputs = self.dataset[key]
            
            sum_all += np.sum(inputs,axis = 0) #Sum over all t
            total_updates += sequenceLength
        
        mean = sum_all/total_updates
        
        return torch.tensor(mean, dtype=torch.float)
    