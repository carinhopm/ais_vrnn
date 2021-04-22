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
shipFileName = 'aisMix_2002'
binedges = (config.LAT_EDGES, config.LON_EDGES, config.SOG_EDGES, config.COG_EDGES)
batch_size = 32

'''
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
     'resampleFrequency': config.RESAMPLEFREQ,
     'timeOffset': config.T_OFFSET
    },
    datasets_path = config.datasets_path,
    dataset_filename = config.datapath
)

with open(config.datasets_path + shipFileName + ".pkl", "wb") as f:
        pickle.dump(tracks, f)
'''
        
##dataset_utils.makeDatasetSplits(shipFileName, '24hour_' + shipFileName) ##Probably need to change this function

class PadSequence:
    def __call__(self, batch):
                
        # each element in "batch" is a tuple ( mmsis,  shiptypes,  lengths, inputs, targets)
        # Get each sequence and pad it
        mmsis = [x[0] for x in batch] # Maritime Mobile Service Identity numbers
        shiptypes = [x[1] for x in batch] # tank, cargo, etc.
        lengths = [x[2] for x in batch] # used as measure of size
        inputs = [x[3] for x in batch] # they are normalized 
        targets = [x[4] for x in batch] # seems to contain the real path of the vessel
                                        # lat, lon, speed, course (NOT NORMALIZED)
                
        inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)

        return  torch.tensor(mmsis),  torch.tensor(shiptypes),  torch.tensor(lengths, dtype=torch.float), inputs_padded, targets_padded

# different lengths (use max/min for dimensions)
trainset = dataset_utils.AISDataset("data/train_CarFishPassTankSailPlea.pkl")
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 0, collate_fn=PadSequence())
testset = dataset_utils.AISDataset("data/test_CarFishPassTankSailPlea.pkl", train_mean = trainset.mean)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 0, collate_fn=PadSequence())

train_n = len(trainset)
test_n = len(testset)
num_batches = len(train_loader)
#num_epochs = ceil(80000/num_batches)
num_epochs = 1

print(len(trainset))
print(len(testset))

mmsi, _, _, _, x = trainset[0]
#plotting.Plot4HotEncodedTrack(x, binedges, ax=None)

model = VRNN.VRNN(input_shape=trainset.datadim, latent_shape=100, generative_bias=trainset.mean, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

load_model = False
if load_model:
    modelName = 'model_' + shipFileName + '_150'
    model.load_state_dict(torch.load('models/saved_models/' + modelName +'.pth', map_location=device))

model.to(device)

def computeLoss(log_px, log_pz, log_qz, lengths, beta=1):
    
    max_len = len(log_px)
    curmask = torch.arange(max_len, device=device)[:, None] < lengths[None, :] #max_seq_len X Batch
    
    log_px = torch.stack(log_px, dim=0) * curmask
    log_px = log_px.sum(dim=0) #Sum over time
   
    log_pz = torch.stack(log_pz, dim=0) * curmask
    log_qz = torch.stack(log_qz, dim=0) * curmask
    kl = log_qz.sum(dim=0) - log_pz.sum(dim=0) #Sum over time
    
    loss = log_px - beta * kl #recon loss - beta_kl
    loss = torch.mean(loss/lengths) #mean over batch
    
    return -loss, log_px, kl

loss_tot = []
kl_tot = []
recon_tot = []
val_loss_tot = []
val_kl_tot = []
val_recon_tot = []
for epoch in range(1, num_epochs+1): #num_epochs+1
    #Begin training loop
    loss_epoch = 0
    kl_epoch = 0
    recon_epoch = 0
    model.train()
    for i, (_, _, lengths, inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)
        
        log_px, log_pz, log_qz, _, _ = model(inputs,targets,logits=None)
        
        loss, log_px, kl = computeLoss(log_px, log_pz, log_qz, lengths)
        
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_epoch += loss.item()*len(lengths)
        kl_epoch += torch.sum(kl/lengths).item()
        recon_epoch += torch.sum(log_px/lengths).item()
    
    loss_tot.append(loss_epoch/train_n)
    kl_tot.append(kl_epoch/train_n)
    recon_tot.append(recon_epoch/train_n)
    
    #Begin validation loop
    val_loss = 0
    val_kl = 0
    val_recon = 0
    model.eval()
    for i, (_, _, lengths, inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)
        
        log_px, log_pz, log_qz, _, _ = model(inputs,targets,logits=None)
        
        loss, log_px, kl = computeLoss(log_px, log_pz, log_qz, lengths)
                
        val_loss += loss.item()*len(lengths)
        val_kl += torch.sum(kl/lengths).item()
        val_recon += torch.sum(log_px/lengths).item()
    
    val_loss_tot.append(val_loss/test_n)
    val_kl_tot.append(val_kl/test_n)
    val_recon_tot.append(val_recon/test_n)
    
    datapoints = np.random.choice(test_n, size = 3, replace=False)
    #plotting.make_vae_plots((loss_tot, kl_tot, recon_tot, val_loss_tot, val_kl_tot, val_recon_tot), model, datapoints, testset, binedges, device)
    
    print('Epoch {} of {} finished. Trainingloss = {}. Validationloss = {}'.format(epoch, num_epochs, loss_epoch/train_n, val_loss/test_n))
    
    if (epoch%10==0):
        torch.save(model.state_dict(), 'models/saved_models/model_' + shipFileName + '_' + str(epoch) + '.pth')
        
        trainingCurves = {
            'loss_tot': loss_tot,
            'kl_tot': kl_tot,
            'recon_tot': recon_tot,
            'val_loss_tot': val_loss_tot,
            'val_kl_tot': val_kl_tot,
            'val_recon_tot': val_recon_tot
        }
        with open('models/saved_models/trainingCurves_' + shipFileName + '.pkl', "wb") as f:
            pickle.dump(trainingCurves, f)
        

trainingCurves = {
    'loss_tot': loss_tot,
    'kl_tot': kl_tot,
    'recon_tot': recon_tot,
    'val_loss_tot': val_loss_tot,
    'val_kl_tot': val_kl_tot,
    'val_recon_tot': val_recon_tot
}

torch.save(model.state_dict(), 'models/model_' + shipFileName + '.pth')
with open('models/trainingCurves_' + shipFileName + '.pkl', "wb") as f:
        pickle.dump(trainingCurves, f)

