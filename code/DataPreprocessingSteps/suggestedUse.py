
## Define design choices
classnames, nClasses = dataset_utils.classNames()
dataType = 'AIS'
area = 'Custom'
startTime = datetime.datetime(2020, 10, 1, 0, 0, 0, 0) #2020-10-1 00:00:00.000
endTime = datetime.datetime(2020, 12, 31, 23, 59, 59, 999) #2020-12-31 23:59:59.999
shiptypes = config.SHIPTYPE_CARGO + config.SHIPTYPE_FISHING + config.SHIPTYPE_PASSENGER +config.SHIPTYPE_TANKER + config.SHIPTYPE_SAILING + config.SHIPTYPE_PLEASURE + config.SHIPTYPE_MILITARY
shipNames = np.unique([dataset_utils.convertShipTypeToName(str(st)) for st in shiptypes])
shortShipNames = ''.join([name[:3] for name in shipNames])
minTrackLength = 7200 #In seconds #Min track time length is 2 hours
maxTrackLength = 7200 #In seconds #Max track length is 2 hours.
resampleFreq = 60 #Time between samples in seconds

#Make a string use to name files with these design choices
shipFileName = dataType + '_' + area + '_' + startTime.strftime('%d%m%Y') + '_' + endTime.strftime('%d%m%Y') + '_' + shortShipNames + '_' + str(minTrackLength) + '_' + str(maxTrackLength) + '_' + str(resampleFreq)

#Run the preprocessing. This of course only needs to run once for each design choice
tracks = createAISdata.createAISdataset(
    {'ROI': (config.LAT_MIN, config.LAT_MAX, config.LON_MIN, config.LON_MAX), 
     'timeperiod': (startTime.timestamp(), endTime.timestamp()), 
     'maxspeed': config.SOG_MAX, 
     'navstatuses': config.MOV_NAV_STATUSES, 
     'shiptypes': shiptypes, 
     'binedges': (config.LAT_EDGES, config.LON_EDGES, config.SOG_EDGES, config.COG_EDGES), 
     'minTrackLength': minTrackLength,
     'maxTrackLength': maxTrackLength, 
     'resampleFrequency': resampleFreq
    }, 
    shipFileName
)

#Make train/test splits randomly
n = len(tracks['indicies'])//5
testIndex = np.random.choice(tracks['indicies'], size=n, replace=False)
trainIndex = [index for index in tracks['indicies'] if index not in testIndex]

tracks['testIndicies'] = testIndex
tracks['trainIndicies'] = trainIndex

with open('data/datasetInfo_' + shipFileName + '.pkl','wb') as file:
    pickle.dump(tracks, file)
    


#Define dataset
batch_size=32

class PadSequence:
    def __call__(self, batch):
                
        # each element in "batch" is a tuple ( mmsis,  shiptypes,  lengths, inputs, targets)
        # Get each sequence and pad it
        mmsis = [x[0] for x in batch]
        shiptypes = [x[1] for x in batch]
        lengths = [x[2] for x in batch]
        inputs = [x[3] for x in batch]
        targets = [x[4] for x in batch]
                
        inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)

        return  torch.tensor(mmsis),  torch.tensor(shiptypes),  torch.tensor(lengths, dtype=torch.float), inputs_padded, targets_padded

trainset = dataset_utils.AISDataset('data/datasetInfo_' + shipFileName + '.pkl')
testset = dataset_utils.AISDataset('data/datasetInfo_' + shipFileName + '.pkl', train_mean = trainset.mean)

train_n = len(trainset)
test_n = len(testset)

train_classnames = [classnames[x] for x in torch.unique(trainset.labels)]
test_classnames = [classnames[x] for x in torch.unique(testset.labels)]

weights = torch.zeros(train_n)
for index in range(train_n):
    weights[index] = 1 / trainset.samples_pr_class[trainset.labels[index]]

sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, train_n,replacement=True)

#Check whether to zero pad to same length 
if trainset.params['minTrackLength']==trainset.params['maxTrackLength']:
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=sampler, pin_memory=True, num_workers = 1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 1)
else:
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=sampler, pin_memory=True, num_workers = 1, collate_fn=PadSequence)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 1, collate_fn=PadSequence)
    
num_batches = len(train_loader)
num_epochs = ceil(80000/num_batches)