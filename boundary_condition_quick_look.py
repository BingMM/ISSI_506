#%%

import matplotlib.pyplot as plt
import h5py

#%% Load boundary condition data

base = '/home/bing/server/GEMINI/hack_version'

timestep = 15990

filename = base + '/inputs/fields/20160303_{}.000000.h5'.format(timestep)
dat_B = {}
with h5py.File(filename, 'r') as f:
    for key in list(f.keys())[:-2]:
        dat_B[key] = f[key][:]

filename = base + '/inputs/fields/simgrid.h5'
with h5py.File(filename, 'r') as f:
    dat_B['mlat'] = f['mlat'][:]
    dat_B['mlon'] = f['mlon'][:]


filename = base + '/inputs/precip/20160303_{}.000000.h5'.format(timestep)
dat_G = {}
with h5py.File(filename, 'r') as f:
    for key in list(f.keys())[:-1]:
        dat_G[key] = f[key][:]

filename = base + '/inputs/fields/simgrid.h5'
with h5py.File(filename, 'r') as f:
    dat_G['mlat'] = f['mlat'][:]
    dat_G['mlon'] = f['mlon'][:]



'''
According to documentation mlat and mlon is magnetic coordinates.
No specification.
'''

#%%

plt.ioff()

fig, axs = plt.subplots(1, 2, figsize=(15, 10))
pc = axs[0].pcolormesh(dat_B['mlon'], dat_B['mlat'], dat_B['Vminx1it'], cmap='bwr')
plt.colorbar(pc)
pc = axs[1].pcolormesh(dat_G['mlon'], dat_G['mlat'], dat_G['Qp'], cmap='magma')
plt.colorbar(pc)
for ax in axs:
    ax.set_title(timestep)
    ax.set_xlabel('mlon')
    ax.set_ylabel('mlon')

plt.savefig('/home/bing/BCSS-DAG Dropbox/Michael Madelaire/work/code/repos/ISSI_506/figures/boundary_condition_quick_look.png', bbox_inches='tight')
plt.close('all')
plt.ion()