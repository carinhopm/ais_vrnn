import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import requests
import datetime
import torch
import os
import importlib
import sys
import re
import pickle
from IPython.display import clear_output
from mpl_toolkits import mplot3d
from io import BytesIO
from math import log, exp, tan, atan, ceil
from PIL import Image

def get_maps_image(NW_lat_long, SE_lat_long, zoom=18):

    # circumference/radius
    tau = 6.283185307179586
    # One degree in radians, i.e. in the units the machine uses to store angle,
    # which is always radians. For converting to and from degrees. See code for
    # usage demonstration.
    DEGREE = tau/360

    ZOOM_OFFSET = 8
    GOOGLE_MAPS_API_KEY = None  # set to 'your_API_key'

    # Max width or height of a single image grabbed from Google.
    MAXSIZE = 640
    # For cutting off the logos at the bottom of each of the grabbed images.  The
    # logo height in pixels is assumed to be less than this amount.
    LOGO_CUTOFF = 32

    def latlon2pixels(lat, lon, zoom):
        mx = lon
        my = log(tan((lat + tau/4)/2))
        res = 2**(zoom + ZOOM_OFFSET) / tau
        px = mx*res
        py = my*res
        return px, py

    def pixels2latlon(px, py, zoom):
        res = 2**(zoom + ZOOM_OFFSET) / tau
        mx = px/res
        my = py/res
        lon = mx
        lat = 2*atan(exp(my)) - tau/4
        return lat, lon

        ullat, ullon = NW_lat_long
        lrlat, lrlon = SE_lat_long

        # convert all these coordinates to pixels
        ulx, uly = latlon2pixels(ullat, ullon, zoom)
        lrx, lry = latlon2pixels(lrlat, lrlon, zoom)

        # calculate total pixel dimensions of final image
        dx, dy = lrx - ulx, uly - lry

        # calculate rows and columns
        cols, rows = ceil(dx/MAXSIZE), ceil(dy/MAXSIZE)

        # calculate pixel dimensions of each small image
        width = ceil(dx/cols)
        height = ceil(dy/rows)
        heightplus = height + LOGO_CUTOFF

        # assemble the image from stitched
        final = Image.new('RGB', (int(dx), int(dy)))
        for x in range(cols):
            for y in range(rows):
                dxn = width * (0.5 + x)
                dyn = height * (0.5 + y)
                latn, lonn = pixels2latlon(
                        ulx + dxn, uly - dyn - LOGO_CUTOFF/2, zoom)
                position = ','.join((str(latn/DEGREE), str(lonn/DEGREE)))
                print(x, y, position)
                urlparams = {
                        'center': position,
                        'zoom': str(zoom),
                        'size': '%dx%d' % (width, heightplus),
                        'maptype': 'satellite',
                        'sensor': 'false',
                        'scale': 1
                    }
                if GOOGLE_MAPS_API_KEY is not None:
                    urlparams['key'] = GOOGLE_MAPS_API_KEY

                url = 'http://maps.google.com/maps/api/staticmap'
                try:                  
                    response = requests.get(url, params=urlparams)
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    print(e)
                    sys.exit(1)

                im = Image.open(BytesIO(response.content))                  
                final.paste(im, (int(x*width), int(y*height)))

        return final

def Plot4HotEncodedTrack(encodedTrack, edges, ax=None):
    
    seq_len, data_dim = encodedTrack.shape
    lat_edges, lon_edges, speed_edges, course_edges = edges
    
    lat_dim = len(lat_edges) - 1
    lon_dim = len(lon_edges) - 1
    speed_dim = len(speed_edges) - 1
    course_dim = len(course_edges) - 1
    
    lat_centers = [round((lat_edges[i]+lat_edges[i+1])/2,3) for i in range(len(lat_edges)-1)] 
    lon_centers = [round((lon_edges[i]+lon_edges[i+1])/2,3) for i in range(len(lon_edges)-1)] 
    speed_centers = [round((speed_edges[i]+speed_edges[i+1])/2,3) for i in range(len(speed_edges)-1)] 
    course_centers = [round((course_edges[i]+course_edges[i+1])/2,3) for i in range(len(course_edges)-1)] 
   
    lat = np.zeros((seq_len))
    lon = np.zeros((seq_len))
    speed = np.zeros((seq_len))
    course = np.zeros((seq_len))
    
    for i in range(seq_len):
        lat[i] = lat_centers[np.argmax(encodedTrack[i,0:lat_dim])]
        lon[i] = lon_centers[np.argmax(encodedTrack[i,lat_dim:(lat_dim+lon_dim)])]
        speed[i] = speed_centers[np.argmax(encodedTrack[i,(lat_dim+lon_dim):(lat_dim+lon_dim+speed_dim)])]
        course[i] = course_centers[np.argmax(encodedTrack[i,(lat_dim+lon_dim+speed_dim):(lat_dim+lon_dim+speed_dim+course_dim)])]
    
    points = np.array([lon, lat, speed]).T.reshape(-1, 1, 3)
    segments_speed = np.concatenate([points[:-1], points[1:]], axis=1)
    points = np.array([lon, lat, [0]*seq_len]).T.reshape(-1, 1, 3)
    segments_0 = np.concatenate([points[:-1], points[1:]], axis=1)
    
    cmap=plt.get_cmap('viridis') #Blue is start, yellow is end
    colors=[cmap(float(ii)/(seq_len-1)) for ii in range(seq_len-1)]    
    
    if ax == None:
        fig = plt.figure()
        ax = plt.axes(projection="3d")
            
        for ii in range(2,seq_len-1):
            segii=segments_speed[ii]
            lii, =ax.plot(segii[:,0],segii[:,1],segii[:,2],color=colors[ii])
            segii=segments_0[ii]
            lii, =ax.plot(segii[:,0],segii[:,1],segii[:,2],'--',color=colors[ii])
            
            lii.set_solid_capstyle('round')
            
        ax.set_xlabel('longitude')
        ax.set_ylabel('latitude')
        ax.set_zlabel('speed m/s')
    else:
        for ii in range(2,seq_len-1):
            segii=segments_speed[ii]
            lii, =ax.plot(segii[:,0],segii[:,1],segii[:,2],color=colors[ii])
            segii=segments_0[ii]
            lii, =ax.plot(segii[:,0],segii[:,1],segii[:,2],'--',color=colors[ii])
            
            lii.set_solid_capstyle('round')
            
        ax.set_xlabel('longitude')
        ax.set_ylabel('latitude')
        ax.set_zlabel('speed m/s')
    
    plt.show

def logitToTrack(logits,edges):
    lat_edges, lon_edges, speed_edges, course_edges = edges
    
    logits = logits.squeeze()
    
    seq_len = logits.shape[0]
    
    lat_dim = len(lat_edges) - 1
    lon_dim = len(lon_edges) - 1
    speed_dim = len(speed_edges) - 1
    course_dim = len(course_edges) - 1
    
    recon = torch.zeros(seq_len,lat_dim+lon_dim+speed_dim+course_dim)
    for t in range(seq_len):
        lat_idx = torch.argmax(logits[t,0:lat_dim])
        lon_idx = torch.argmax(logits[t,lat_dim:(lat_dim+lon_dim)])
        speed_idx = torch.argmax(logits[t,(lat_dim+lon_dim):(lat_dim+lon_dim+speed_dim)])
        course_idx = torch.argmax(logits[t,(lat_dim+lon_dim+speed_dim):(lat_dim+lon_dim+speed_dim+course_dim)])

        recon[t,lat_idx] = 1
        recon[t,lat_dim+lon_idx] = 1
        recon[t,lat_dim+lon_dim+speed_idx] = 1
        recon[t,lat_dim+lon_dim+speed_dim+course_idx] = 1
    
    return recon    

def plot_recon(datapoint, binedges, model, device):
     
    fig, ax = plt.subplots(1,2,figsize=(20,10))
    
    _, _, length, datainput, datatarget = datapoint
    datadim = datainput.shape[1]
    datainput = datainput.to(device)
    datatarget_gpu = datatarget.to(device)
    
    logits = torch.zeros(length.int().item(),1,datadim, device = device)
    _, _, _, logits, _ = model(datainput.unsqueeze(0), datatarget_gpu.unsqueeze(0), logits=logits)
        
    logits = logits.cpu()
    recon = logitToTrack(logits, binedges)
        
    ax[0].remove()
    ax[0]=fig.add_subplot(1,2,1,projection='3d')
    Plot4HotEncodedTrack(datatarget, binedges, ax[0])
        
    ax[1].remove()
    ax[1]=fig.add_subplot(1,2,2,projection='3d')
    Plot4HotEncodedTrack(recon, binedges, ax[1])

def make_vae_plots(losses, model, datapoints, validationdata, binedges, device):
    
    clear_output(wait=True)
    
    loss_tot, kl_tot, recon_tot, val_loss_tot, val_kl_tot, val_recon_tot = losses
    
    fig, ax = plt.subplots(3,3,figsize=(20,20))
    
    ax[0,0].plot(loss_tot, label='Training Loss')
    ax[0,0].plot(val_loss_tot, label='Validation Loss')
    ax[0,0].set_title('Loss')
    ax[0,0].set_xlabel('Epoch')
    ax[0,0].legend()
    
    ax[0,1].plot(kl_tot, label='Training KL-divergence')
    ax[0,1].plot(val_kl_tot, label='Validation KL-divergence')
    ax[0,1].set_title('KL divergence')
    ax[0,1].set_xlabel('Epoch')
    ax[0,1].legend()
    
    ax[0,2].plot(recon_tot, label='Training Reconstruction')
    ax[0,2].plot(val_recon_tot, label='Validation Reconstruction')
    ax[0,2].set_title('Reconstruction probability log_prob(x)')
    ax[0,2].set_xlabel('Epoch')
    ax[0,2].legend()
    
    for i, idx in enumerate(datapoints):
        _, _, length, datainput, datatarget = validationdata[idx]
        datainput = datainput.to(device)
        datatarget_gpu = datatarget.to(device)
        
        logits = torch.zeros(length.int().item(),1,validationdata.datadim, device = device)
        _, _, _, logits, _ = model(datainput.unsqueeze(0), datatarget_gpu.unsqueeze(0),logits=logits)
        
        logits = logits.cpu()
        recon = logitToTrack(logits, binedges)
        
        ax[1,i].remove()
        ax[1,i]=fig.add_subplot(3,3,4+i,projection='3d')
        Plot4HotEncodedTrack(datatarget, binedges, ax[1,i])
        
        ax[2,i].remove()
        ax[2,i]=fig.add_subplot(3,3,7+i,projection='3d')
        Plot4HotEncodedTrack(recon, binedges, ax[2,i])
        
    plt.show()