#%% Import

import numpy as np
import matplotlib.pyplot as plt
from secsy import cubedsphere
import lompe
import dipole
import pickle
from scipy.interpolate import griddata

#%% Paths

p_output = '/home/bing/BCSS-DAG Dropbox/Michael Madelaire/work/code/repos/ISSI_506/'

#%% Hyper paramters

RE = 6371.2 # Earth radius in km

#%% Load the cubed sphere grid

with open(p_output + 'data/cubedsphere_grid.pkl', 'rb') as f:
    grid = pickle.load(f)

#%% Load the gridded data
with open(p_output + 'data/gridded_data.pkl', 'rb') as f:
    data = pickle.load(f)   

#%% Get old grid settings

glon_c, glat_c = grid.projection.position
L, W = grid.L, grid.W
Lres, Wres = grid.Lres, grid.Wres

#%% Define dipole

# According to the GEMINI docs, this is the centered dipole they use in GEMINI: 
# https://zenodo.org/record/3903830/files/GEMINI.pdf
glon_pole   = 290.24
glat_pole   = 90 - 11.435
M           = 7.94e22 # magnetic moment in A/m^2
mu0         = 4*np.pi*10**(-7)
B0          = mu0*M/(4*np.pi*(RE*1e3)**3) # from e.g. https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/RG025i001p00001
dp          = dipole.Dipole(dipole_pole=(glat_pole, glon_pole), B0=B0)

#%% Define new grid parameters

# shift in dipole coordinates
lat_swift = -6
lon_swift = 1

# Current grid center in centered dipole coordinates
mlat_c, mlon_c = dp.geo2mag(glat_c, glon_c)

# New grid center in centered dipole coordinates
mlat_c_new = mlat_c + lat_swift
mlon_c_new = mlon_c + lon_swift

# New grid center in geo
glat_c_new, glon_c_new = dp.mag2geo(mlat_c_new, mlon_c_new)

# Define new grid
position = (glon_c_new, glat_c_new)
orientation = np.degrees(lompe.data_tools.dataloader.getbearing(np.array([glat_c_new]), 
                                                                np.array([glon_c_new]), 
                                                                np.array([glat_pole]), 
                                                                np.array([glon_pole])))

grid_new = cubedsphere.CSgrid(cubedsphere.CSprojection(position, -orientation[0]),
                             grid.L, grid.W, grid.Lres, grid.Wres, R=grid.R)

#%% Save new grid

filename = p_output + 'data/grid_new.pkl'
with open(filename, 'wb') as f:
    pickle.dump(grid_new, f)

#%% 

# Convert current into magnetic
mlat, mlon, mJe, mJn         = dp.geo2mag(grid.lat, grid.lon, data['Je'], data['Jn'])
mlat, mlon, mJperpe, mJperpn = dp.geo2mag(grid.lat, grid.lon, data['Jperpe'], data['Jperpn'])

# Move it
mlat += lat_swift
mlon += lon_swift

# Convert back to geo
glat, glon, Je, Jn          = dp.mag2geo(mlat, mlon, mJe, mJn)
glat, glon, Jperpe, Jperpn  = dp.mag2geo(mlat, mlon, mJperpe, mJperpn)

# interpolate to new grid
glon[glon>180] -= 360
points = np.vstack((glon.flatten(), glat.flatten())).T
points_new = np.vstack((grid_new.lon.flatten(), grid_new.lat.flatten())).T
data_new = {}
data_new['Je']      = griddata(points, Je.flatten(), points_new, method='linear').reshape(grid.shape)
data_new['Jn']      = griddata(points, Jn.flatten(), points_new, method='linear').reshape(grid.shape)
data_new['Ju']      = griddata(points, data['Ju'].flatten(), points_new, method='linear').reshape(grid.shape)
data_new['Jperpe']  = griddata(points, Jperpe.flatten(), points_new, method='linear').reshape(grid.shape)
data_new['Jperpn']  = griddata(points, Jperpn.flatten(), points_new, method='linear').reshape(grid.shape)
data_new['Jperpu']  = griddata(points, data['Jperpu'].flatten(), points_new, method='linear').reshape(grid.shape)
data_new['SH']      = griddata(points, data['SH'].flatten(), points_new, method='linear').reshape(grid.shape)
data_new['SP']      = griddata(points, data['SP'].flatten(), points_new, method='linear').reshape(grid.shape)

#%% Save new data

filename = p_output + 'data/data_new.pkl'
with open(filename, 'wb') as f:
    pickle.dump(data_new, f)

#%% Plot comparisons of the old and new grid 

# Load ground magnetometer locations
with open(p_output + 'data/supermag_stations.pkl', 'rb') as f:
    stations = pickle.load(f)

# Start plot
plt.ioff()
fig, axs = plt.subplots(1, 2, figsize=(15, 9))

for (ax, gr, dat) in zip(axs, [grid, grid_new], [data, data_new]):
    
    # Coastline
    for cl in gr.projection.get_projected_coastlines():
        xi, eta = cl
        ax.plot(xi, eta, color='k', linewidth=.7)
        ax.plot(xi, eta, color='cyan', linewidth=.5)
    
    # FAC
    f = np.isfinite(dat['Ju'])
    vmax = np.nanmax(abs(dat['Ju']))
    ax.tricontourf(gr.xi[f], gr.eta[f], dat['Ju'][f], 
                   cmap='bwr', levels=np.linspace(-vmax, vmax, 40))
    
    # Plot magnetometers
    xi, eta = gr.projection.geo2cube(stations['GEOLON'], stations['GEOLAT'])
    ax.plot(xi, eta, '*', markersize=8, markerfacecolor='cyan', 
            markeredgecolor='k')
    
    # Magnetic pole
    xi, eta = gr.projection.geo2cube(glon_pole, glat_pole)
    ax.plot(xi, eta, 's', markersize=8, markerfacecolor='tab:blue', markeredgecolor='k')
    
    # Rotational axis pole
    xi, eta = gr.projection.geo2cube(0, 90)
    ax.plot(xi, eta, 's', markersize=8, markerfacecolor='tab:red', markeredgecolor='k')
    
    # Magnetic grid
    for i in range(40, 90, 5):
        mlat, mlon = dp.mag2geo(np.ones(1000)*i, np.linspace(0, 360, 1000))
        xi, eta = gr.projection.geo2cube(mlon, mlat)
        ax.plot(xi, eta, color='gray', linewidth=.3)

    for i in range(0, 360, 15):
        mlat, mlon = dp.mag2geo(np.linspace(0, 90, 1000), np.ones(1000)*i)
        xi, eta = gr.projection.geo2cube(mlon, mlat)
        ax.plot(xi, eta, color='gray', linewidth=.3)    

    # Highlight 60-70 mlat
    for i in range(60, 80, 10):
        mlat, mlon = dp.mag2geo(np.ones(1000)*i, np.linspace(0, 360, 1000))
        xi, eta = gr.projection.geo2cube(mlon, mlat)
        ax.plot(xi, eta, color='w', linewidth=.3)
        ax.plot(xi, eta, '--', color='k', linewidth=1.5)
    
    # Small adjustment
    ax.set_ylim([gr.eta.min()*0.6, gr.eta.max()*1.05])
    ax.set_xlim([gr.xi.min()*0.6, gr.xi.max()*0.6])
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_aspect('equal')
    
axs[0].set_title('Old grid', fontsize=20)
axs[1].set_title('New grid', fontsize=20)

# Save
plt.savefig(p_output + 'figures/grid_comparison.png', bbox_inches='tight')
plt.close('all')
plt.ion()




















