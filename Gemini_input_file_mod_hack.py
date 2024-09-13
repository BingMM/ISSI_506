#%% Import

import numpy as np
import matplotlib.pyplot as plt
import h5py
import copy

#%% hyper parameters

base = '/home/bing/BCSS-DAG Dropbox/Data/issi_team_506_gemini_run/aurora_EISCAT3D/inputs/'



dlat = 0.03383255
lat_swift = 3.3
id_swift = int(lat_swift / dlat)

center_id = 574

ts = np.arange(15300, 16500+10, 10)

filename = base + 'fields_mod/simgrid.h5'
with h5py.File(filename, 'r') as f:
    mlat = f['mlat'][...]
    mlon = f['mlon'][...]

# 180 km mlat limits
mlat_bot = 58.77
mlat_top = 85.92

mlat_bot_id = np.argmin(abs(mlat-mlat_bot))
mlat_top_id = np.argmin(abs(mlat-mlat_top))

#%% Define Gauss drop-off

# id_start = 100 # hack version 1-4
id_start = 150 # hack version 5

nstd = 5
std = id_start/nstd
x = np.arange(id_start+1)
y = 1/np.sqrt(2*np.pi*std**2)*np.exp(-(x-id_start)**2/(2*std**2))
y /= np.max(y)

# mask mlon
mask_mlon = np.ones((1024, 1024)) # Left right

mask_side = np.tile(y, (1024, 1))

mask_mlon[:, :id_start+1] = mask_side
mask_mlon[:, -id_start-1:] = np.flip(mask_side, axis=1)

# mask mlat
mask_mlat = np.ones((1024, 1024)) # Top bot

mask_mlat[:mlat_bot_id, :] = 0
mask_mlat[mlat_bot_id:mlat_bot_id+id_start+1, :] = mask_side.T

mask_mlat[mlat_top_id:, :] = 0
mask_mlat[mlat_top_id-id_start-1:mlat_top_id, :] = np.flip(mask_side.T, axis=0)

# Create mask
f = mask_mlon < mask_mlat
mask = copy.deepcopy(mask_mlat)
mask[f] = mask_mlon[f]

#%% FAC mods

for t in ts:
    print(t)
    
    filename = base + 'fields_mod/20160303_{}.000000.h5'.format(str(t))
    f = h5py.File(filename, 'r+')
    
    # Grab data
    var = f['Vminx1it'][:]
    
    # Center it
    var = np.hstack((var[:, :center_id],
                     -np.flip(var[:, :center_id], axis=1)))
    var = var[:, 62:-62]

    # Move it up
    var = np.vstack((np.tile(var[0, :], (id_swift, 1)), var))
    var = var[:-id_swift, :]

    # Shove it back in
    f['Vminx1it'][...] = var*mask
    
    f.close()


#%% Conductance mods

for t in ts:
    print(t)
    
    filename = base + 'precip_mod/20160303_{}.000000.h5'.format(str(t))
    f = h5py.File(filename, 'r+')
    
    # Grab data
    var = f['Qp'][:]
    
    # Center it
    var = np.hstack((var, 
                     np.tile(var[:, -1], (124, 1)).T))
    var = var[:, 62:-62]

    # High conductance at upward FAC
    var = np.flip(var, axis=1)

    # Move it up
    var = np.vstack((np.tile(var[0, :], (id_swift, 1)), var))
    var = var[:-id_swift, :]

    # Shove it back in
    f['Qp'][...] = var*mask
    
    f.close()

#%%

fig, axs = plt.subplots(1, 2, figsize=(15, 9))

filename = base + 'fields_mod/20160303_16000.000000.h5'
with h5py.File(filename, 'r') as f:    
    axs[0].imshow(f['Vminx1it'][...], cmap='bwr', origin='lower')

filename = base + 'precip_mod/20160303_16000.000000.h5'
with h5py.File(filename, 'r') as f:    
    axs[1].imshow(f['Qp'][...], cmap='magma', origin='lower')

