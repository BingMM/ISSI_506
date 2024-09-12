#%% Import

import sys
import numpy as np
import matplotlib.pyplot as plt
from secsy import cubedsphere
import lompe
import gemini3d.read
import dipole
import polplot
import pickle
from tqdm import tqdm
from gemini3d.grid.gridmodeldata import model2pointsgeogcoords
from polplot import Polarplot

sys.path.append('/home/bing/BCSS-DAG Dropbox/Michael Madelaire/work/code/repos/ISSI_506')
import helpers

#%% Paths

p_gemini = '/home/bing/server/GEMINI/hack_version/'
p_output = '/home/bing/BCSS-DAG Dropbox/Michael Madelaire/work/code/repos/ISSI_506/'

#%% Load the cubed sphere grid

with open(p_output + 'data/cubedsphere_grid.pkl', 'rb') as f:
    grid = pickle.load(f)
    
#%% Load GEMINI simulation

timestep    = 15980
timesteps   = np.arange(15300, 16500+10, 10)
timestep_id = np.argmin(abs(timesteps - timestep))

cfg = gemini3d.read.config(p_gemini)
dat = gemini3d.read.frame(p_gemini, cfg['time'][timestep_id])
xg  = gemini3d.read.grid(p_gemini)

#%% Calculate desired quantities

dat = helpers.read_gemini(xg, dat)

#%% Calculate height integrated values

#alts = np.concatenate((np.arange(90,140,2),np.arange(140,170,5),np.arange(170,230,10),np.arange(230,830,50)))*1e3

alt_step = 1
#alts = np.arange(90, 830+alt_step, alt_step)*1e3
alts = np.arange(90, 2000+alt_step, alt_step)*1e3
lats = grid.lat.flatten()
lons = grid.lon.flatten()

Je = np.zeros(grid.shape)
Jn = np.zeros(grid.shape)
Jperpe = np.zeros(grid.shape)
Jperpn = np.zeros(grid.shape)
SH = np.zeros(grid.shape)
SP = np.zeros(grid.shape)

for i, alt_i in tqdm(enumerate(alts), 
                     desc='Calculating one shell at a time',
                     total=alts.size):
    
    Je += model2pointsgeogcoords(xg, dat['je'], np.ones(grid.size)*alt_i, 
                                 lons, lats).reshape(grid.shape)*alt_step*1e3
    Jn += model2pointsgeogcoords(xg, dat['jn'], np.ones(grid.size)*alt_i, 
                                 lons, lats).reshape(grid.shape)*alt_step*1e3
    Jperpe += model2pointsgeogcoords(xg, dat['jperpe'], np.ones(grid.size)*alt_i, 
                                 lons, lats).reshape(grid.shape)*alt_step*1e3
    Jperpn += model2pointsgeogcoords(xg, dat['jperpn'], np.ones(grid.size)*alt_i, 
                                 lons, lats).reshape(grid.shape)*alt_step*1e3
    SH += model2pointsgeogcoords(xg, dat['sh'], np.ones(grid.size)*alt_i, 
                                 lons, lats).reshape(grid.shape)*alt_step*1e3
    SP += model2pointsgeogcoords(xg, dat['sp'], np.ones(grid.size)*alt_i, 
                                 lons, lats).reshape(grid.shape)*alt_step*1e3

#%% Calculate FACs at 180 km

Jperpu = model2pointsgeogcoords(xg, dat['jperpu'], np.ones(grid.size)*180*1e3, lons, lats).reshape(grid.shape)

Ju = model2pointsgeogcoords(xg, dat['ju'], np.ones(grid.size)*180*1e3, lons, lats).reshape(grid.shape)

#%% Plot the height integrated quantities

# Height integrated Hall conductivity
plt.ioff()
fig = plt.figure(figsize=(10, 10))
ax = plt.gca()
pax = Polarplot(ax)
cc = pax.tricontourf(grid.lat, grid.lon/15, SH, cmap='magma', levels=np.linspace(0, np.nanmax(SH), 40))
plt.colorbar(cc)
plt.savefig(p_output + 'figures/height_integrated_SH.png', bbox_inches='tight')
plt.close('all')
plt.ion()

# Height integrated Pedersen conductivity
plt.ioff()
fig = plt.figure(figsize=(10, 10))
ax = plt.gca()
pax = Polarplot(ax)
cc = pax.tricontourf(grid.lat, grid.lon/15, SP, cmap='magma', levels=np.linspace(0, np.nanmax(SP), 40))
plt.colorbar(cc)
plt.savefig(p_output + 'figures/height_integrated_SP.png', bbox_inches='tight')
plt.close('all')
plt.ion()

# Height integrated Hall conductivity
plt.ioff()
fig = plt.figure(figsize=(10, 10))
f = np.isfinite(SH)
cc = plt.tricontourf(grid.xi[f], grid.eta[f], SH[f], cmap='magma', levels=np.linspace(0, np.nanmax(SH), 40))
plt.colorbar(cc)
plt.savefig(p_output + 'figures/height_integrated_SH_CS.png', bbox_inches='tight')
plt.close('all')
plt.ion()

# Height integrated Pedersen conductivity
plt.ioff()
fig = plt.figure(figsize=(10, 10))
f = np.isfinite(SH)
cc = plt.tricontourf(grid.xi[f], grid.eta[f], SP[f], cmap='magma', levels=np.linspace(0, np.nanmax(SP), 40))
plt.colorbar(cc)
plt.savefig(p_output + 'figures/height_integrated_SP_CS.png', bbox_inches='tight')
plt.close('all')
plt.ion()

# Height integrated horizontal current
plt.ioff()
fig = plt.figure(figsize=(10, 10))
ax = plt.gca()
pax = Polarplot(ax)
ss = 10
pax.quiver(grid.lat[::ss, ::ss].flatten(), grid.lon[::ss, ::ss].flatten()/15, 
           Jn[::ss, ::ss].flatten(), Je[::ss, ::ss].flatten(), 
           scale=5e1, width=5e-4)
plt.savefig(p_output + 'figures/height_integrated_J.png', bbox_inches='tight')
plt.close('all')
plt.ion()

# Height integrated horizontal current on cubed sphere grid
plt.ioff()
fig = plt.figure(figsize=(10, 10))
ss = 10
Jxi, Jeta = grid.projection.vector_cube_projection(Je, Jn, grid.lon, grid.lat,
                                                   return_xi_eta=False)
Jxi = Jxi.reshape(grid.shape)
Jeta = Jeta.reshape(grid.shape)
plt.quiver(grid.xi[::ss, ::ss].flatten(), grid.eta[::ss, ::ss].flatten(), 
           Jxi[::ss, ::ss].flatten(), Jeta[::ss, ::ss].flatten(), 
           scale=3e1, width=1e-3)
plt.savefig(p_output + 'figures/J_CS.png', bbox_inches='tight')
plt.close('all')
plt.ion()

# Ju at 180 km
plt.ioff()
fig = plt.figure(figsize=(10, 10))
ax = plt.gca()
pax = Polarplot(ax)
cc = pax.tricontourf(grid.lat, grid.lon/15, Ju, cmap='bwr', levels=40)
plt.colorbar(cc)
plt.savefig(p_output + 'figures/Ju.png', bbox_inches='tight')
plt.close('all')
plt.ion()

# Jperpu at 180 km (the up component resulting from the 3D current)
plt.ioff()
fig = plt.figure(figsize=(10, 10))
ax = plt.gca()
pax = Polarplot(ax)
cc = pax.tricontourf(grid.lat, grid.lon/15, Jperpu, cmap='bwr', levels=40)
plt.colorbar(cc)
plt.savefig(p_output + 'figures/Jperpu.png', bbox_inches='tight')
plt.close('all')
plt.ion()

#%% Save height integrated quantities and FAC

data = {'SH': SH, 'SP': SP, 
        'Je': Je, 'Jn': Jn, 'Ju': Ju, 
        'Jperpe': Jperpe, 'Jperpn': Jperpn, 'Jperpu': Jperpu}

with open(p_output + 'data/gridded_data.pkl', 'wb') as f:
    pickle.dump(data, f)
