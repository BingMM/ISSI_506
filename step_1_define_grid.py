#%% Import

import numpy as np
import matplotlib.pyplot as plt
from secsy import cubedsphere
import lompe
import gemini3d.read
import dipole
import polplot
import pickle

#%% Hyper parameters

target_altitude = 110

#%% Import GEMINI grid

xg = gemini3d.read.grid('/home/bing/BCSS-DAG Dropbox/Michael Madelaire/work/temp_storage/hack_version/inputs/simgrid.h5')

alt_id = np.argmin(abs(xg['alt'][:, 0, 0] - target_altitude*1e3))

#%%

fig = plt.figure()
ax = plt.gca()
pax = polplot.Polarplot(ax)
pax.scatter(xg['glat'][alt_id, :, :].flatten(), xg['glon'][alt_id, :, :].flatten()/15)

#%% Quick visualization of GEMINI grid

plt.ioff()

fig, axs = plt.subplots(1, 2, figsize=(15, 9))
paxs = [polplot.Polarplot(axs[0]), polplot.Polarplot(axs[1])]
paxs[0].scatter(xg['glat'][alt_id, :, :].flatten(),
                xg['glon'][alt_id, :, :].flatten()/15)
axs[0].set_title('GEMINI : Geo : {} km'.format(target_altitude))

paxs[1].plot(90-xg['theta'][alt_id, :, :].flatten()/np.pi*180,
             xg['phi'][alt_id, :, :].flatten()/np.pi*180/15)
axs[1].set_title('GEMINI : Mag : {} km'.format(target_altitude))

plt.savefig('/home/bing/BCSS-DAG Dropbox/Michael Madelaire/work/code/repos/ISSI_506/figures/GEMINI_grid.png', bbox_inches='tight')
plt.close('all')
plt.ion()

#%% Define cubedsphere grid

# Cubedsphere grid center
glat_c, glon_c = 72, 352
position = (glon_c, glat_c)

# Orientation towards magnetic pole
glat_pole, glon_pole = 90 - 11.435, 290.24 # From Jone's code who got it from Matt's code...
orientation = np.degrees(lompe.data_tools.dataloader.getbearing(np.array([glat_c]), 
                                                                np.array([glon_c]), 
                                                                np.array([glat_pole]), 
                                                                np.array([glon_pole])))

# Other grid parameters
L, W = 8000*1e3, 4000*1e3
Lres, Wres = 10*1e3, 10*1e3
RE = 6371.2*1e3
RI = RE + 110*1e3

# The grid
grid = cubedsphere.CSgrid(cubedsphere.CSprojection(position, -orientation[0]),
                          L, W, Lres, Wres, R=RI)

#%% Is this dipole correct? Yes

dip = dipole.Dipole(dipole_pole=(glat_pole, glon_pole))
glat, glon = dip.mag2geo(90-xg['theta'][alt_id, :, :].flatten()/np.pi*180, 
                         xg['phi'][alt_id, :, :].flatten()/np.pi*180)

plt.ioff()

plt.figure(figsize=(10, 10))
ax = plt.gca()
pax = polplot.Polarplot(ax)
pax.scatter(xg['glat'][alt_id, :, :].flatten(),
            xg['glon'][alt_id, :, :].flatten()/15, label='Geo')
pax.scatter(glat, glon/15, alpha=.1, label='Mag->Geo')

plt.title('GEMINI Mag -> Geo via Dipole @ {} km'.format(target_altitude))
plt.legend()

plt.savefig('/home/bing/BCSS-DAG Dropbox/Michael Madelaire/work/code/repos/ISSI_506/figures/GEMINI_grid_conversion_test.png', bbox_inches='tight')
plt.close('all')
plt.ion()

#%% Quick visual comparison between the GEMINI and cubedsphere grid

plt.ioff()

fig = plt.figure(figsize=(10, 10))
ax = plt.gca()
pax = polplot.Polarplot(ax)

pax.scatter(xg['glat'][alt_id, :, :].flatten(),
            xg['glon'][alt_id, :, :].flatten()/15)
pax.scatter(grid.lat_mesh.flatten(), grid.lon_mesh.flatten()/15, alpha=.01)
pax.scatter(grid.projection.position[1], grid.projection.position[0]/15)

plt.savefig('/home/bing/BCSS-DAG Dropbox/Michael Madelaire/work/code/repos/ISSI_506/figures/GEMINI_cubedsphere_grid_comparison.png', bbox_inches='tight')
plt.close('all')
plt.ion()

#%% Save the grid

with open('/home/bing/BCSS-DAG Dropbox/Michael Madelaire/work/code/repos/ISSI_506/data/cubedsphere_grid.pkl', 'wb') as f:
    pickle.dump(grid, f)

