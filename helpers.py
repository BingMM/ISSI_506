#%% import

import numpy as np
import xarray as xr
from gemini3d.grid.gridmodeldata import model2pointsgeogcoords, geog2dipole
from gemini3d.grid.convert import unitvecs_geographic

#%% Hyper parameters

RE = 6371.2*1e3

#%%

def read_gemini(xg, dat):
    
    dat = calc_conductivities(xg, dat)
    dat = compute_enu_components(xg, dat)
    
    return dat

#%%

def calc_conductivities(xg, dat):
    '''
    Function that calculates the Hall and Pedersen conductivities used in GEMINI
    based on the currents and E-fields provided in the output files. This function
    also stores the E-field values in the native GEMINI grid in the dat structure.
    '''
    
    dat = gemini_gradient(xg, dat, q='Phitop')
    shape = dat.J1.shape
    E2 = -dat.gradPhitop_x2
    E3 = -dat.gradPhitop_x3
    Emag = np.sqrt(E2**2 + E3**2)
    ehat = np.stack((np.zeros(shape), E2/Emag, E3/Emag))
    j_ped_mag = dat.J2 * ehat[1,:,:,:] + dat.J3 * ehat[2,:,:,:]
    sp = j_ped_mag / Emag
    j_hall_mag = -dat.J2*ehat[2,:,:,:] + dat.J3*ehat[1,:,:,:]
    sh = j_hall_mag / Emag
    
    dat['sp'] = xr.DataArray(sp, dims=('x1','x2','x3'))
    dat['sh'] = xr.DataArray(sh, dims=('x1','x2','x3'))
    dat['E1'] = xr.DataArray(np.zeros(E2.shape), dims=('x1','x2','x3'))
    dat['E2'] = xr.DataArray(E2, dims=('x1','x2','x3'))
    dat['E3'] = xr.DataArray(E3, dims=('x1','x2','x3'))
    
    return dat

#%%

def gemini_gradient(xg, dat, q='Phitop'):
    '''
    Compute the gradient of a scalar field, e.g. electric potential defined in GEMINI's
    curvlinear coordinates. Input arrays must be 2D or 3D arrays
    
    q: quantity to differentiate
    
    '''
    
    #Metric factors defined in eqs 114-116 in GEMINI documentation
    # h1 = xg['r']**3/(RE**2*np.sqrt(1+3*(np.cos(xg['theta']))**2))
    h2 = RE*(np.sin(xg['theta']))**3/np.sqrt(1+3*(np.cos(xg['theta']))**2)
    h3 = xg['r'] * np.sin(xg['theta'])
    
    ndim = len(dat.Phitop.shape)

    if ndim == 2:
        x2x2, x3x3 = np.meshgrid(xg['x2i'][1:], xg['x3i'][1:], indexing='ij')
        q2 = 1/h2 * diff2d(x2x2, dat[q].values, axis=0)
        q3 = 1/h3 * diff2d(x3x3, dat[q].values, axis=1)
        dat['grad'+q+'_x2'] = xr.DataArray(q2, dims=('x1','x2','x3'))
        dat['grad'+q+'_x3'] = xr.DataArray(q3, dims=('x1','x2','x3'))
    if ndim ==3:
        print('Not implemented')
        print(1/0)
    
    return dat

#%%

def diff2d(_x, _y, axis=0):
    '''
    Compute derivatives with central differencing on a 2D mesh grid, but in the
    direction specified with the axis keyword. 
        
    IMPROVE DOCUMENTATION


    Parameters
    ----------
    _x : TYPE
        DESCRIPTION.
    _y : TYPE
        DESCRIPTION.
    axis : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    _derivative : TYPE
        DESCRIPTION.
    '''
    _derivative = np.zeros(_x.shape)
    # Compute central difference for interior points
    if axis == 0:
        I = _x.shape[0]
        for i in range(1, I - 1):
            _derivative[i,:] = (_y[i + 1,:] - _y[i - 1,:]) / (_x[i + 1,:] - _x[i - 1,:])
        # Compute one-sided differences for the edges
        _derivative[0,:] = (_y[1,:] - _y[0,:]) / (_x[1,:] - _x[0,:])  # One-sided difference at the left edge
        _derivative[-1,:] = (_y[-1,:] - _y[-2,:]) / (_x[-1,:] - _x[-2,:])  # One-sided difference at the right edge
    elif axis == 1:
        I = _x.shape[1]
        for i in range(1, I - 1):
            _derivative[:,i] = (_y[:,i + 1] - _y[:,i - 1]) / (_x[:,i + 1] - _x[:,i - 1])
        # Compute one-sided differences for the edges
        _derivative[:,0] = (_y[:,1] - _y[:,0]) / (_x[:,1] - _x[:,0])  # One-sided difference at the left edge
        _derivative[:,-1] = (_y[:,-1] - _y[:,-2]) / (_x[:,-1] - _x[:,-2])  # One-sided difference at the right edge
    else:
        print('Not implemented')
        print(1/0)
    return _derivative


#%%

def compute_enu_components(xg, dat):
    """
    Add ENU components (geographic) of V, J and B to xarray dataset
    """
    
    # Convert velocity to grographic components, use ENU notation
    vu, ve, vn = model_vec2geo_vec(xg, dat, param='v')
    vperpu, vperpe, vperpn = model_vec2geo_vec(xg, dat, param='v', perp=True)
    
    # Convert current to grographic components, use ENU notation
    jperpu, jperpe, jperpn = model_vec2geo_vec(xg, dat, param='J', perp=True)    
    ju, je, jn = model_vec2geo_vec(xg, dat, param='J')
    
    # Convert electric field to grographic components, use ENU notation
    Eu, Ee, En = model_vec2geo_vec(xg, dat, param='E', perp=True)
    
    # Convert magnetic field to grographic components, use ENU notation
    Bu, Be, Bn = model_vec2geo_vec(xg, dat, param='Bmag')

    # Add to data structure
    dat['Ee'] = Ee
    dat['En'] = En
    dat['Eu'] = Eu
    
    dat['ve'] = ve
    dat['vn'] = vn
    dat['vu'] = vu
    dat['vperpe'] = vperpe
    dat['vperpn'] = vperpn
    dat['vperpu'] = vperpu
    
    dat['je'] = je
    dat['jn'] = jn
    dat['ju'] = ju
    dat['jperpe'] = jperpe
    dat['jperpn'] = jperpn
    dat['jperpu'] = jperpu
    
    dat['Be'] = xr.DataArray(Be, dims=('x1','x2','x3'))
    dat['Bn'] = xr.DataArray(Bn, dims=('x1','x2','x3'))
    dat['Bu'] = xr.DataArray(Bu, dims=('x1','x2','x3'))
    
    return dat

#%%

def model_vec2geo_vec(xg, dat, param='v', perp=False):
    '''
    Function to convert model vector components into geographic conponents. 
    Code provided by M. Zettergren, and put into this function by JPR.

    Parameters
    ----------
    param : 'str'
        'v' (default) or 'J', refering to velocity or current density
    perp : Boolean
        Specifies if only the perpendicular component (2 and 3) of param is to
        be projected to (r, theta, phi) components. Default is False.
        
    Returns
    -------
    (radial, east, north) geographic components of velocity

    '''
    
    [egalt,eglon,eglat]=unitvecs_geographic(xg)     #up, east, north
    #^ returns a set of geographic unit vectors on xg; these are in ECEF geomag
    # comps like all other unit vectors in xg

    # each of the components in models basis projected onto geographic unit 
    # vectors
    if (param != 'Bmag') and perp:
        vgalt=(np.sum(xg["e2"] * egalt, 3) * dat[param + "2"]+ 
               np.sum(xg["e3"] * egalt, 3) * dat[param + "3"])
        
        vglat=(np.sum(xg["e2"] * eglat, 3) * dat[param + "2"]+
               np.sum(xg["e3"] * eglat, 3) * dat[param + "3"])
        
        vglon=(np.sum(xg["e2"] * eglon, 3) * dat[param + "2"]+ 
               np.sum(xg["e3"] * eglon, 3) * dat[param + "3"])
        
    elif (param != 'Bmag'):
        vgalt=(np.sum(xg["e1"] * egalt, 3) * dat[param + "1"]+ 
               np.sum(xg["e2"] * egalt, 3) * dat[param + "2"]+ 
               np.sum(xg["e3"] * egalt, 3) * dat[param + "3"])
        
        vglat=(np.sum(xg["e1"] * eglat, 3) * dat[param + "1"]+ 
               np.sum(xg["e2"] * eglat, 3) * dat[param + "2"]+
               np.sum(xg["e3"] * eglat, 3) * dat[param + "3"])
        
        vglon=(np.sum(xg["e1"] * eglon, 3) * dat[param + "1"]+ 
               np.sum(xg["e2"] * eglon, 3) * dat[param + "2"]+ 
               np.sum(xg["e3"] * eglon, 3) * dat[param + "3"])
        
    else:
        vgalt = np.sum(xg["e1"] * egalt * xg['Bmag'][...,np.newaxis], 3)
        
        vglat = np.sum(xg["e1"] * eglat * xg['Bmag'][...,np.newaxis], 3)
        
        vglon = np.sum(xg["e1"] * eglon * xg['Bmag'][...,np.newaxis], 3)
        
    return vgalt, vglon, vglat # (up, east, north)








#%%

def sample_gemini(xg, dat, poss):
    '''
    Populate data object with observations sampled from GEMINI at locations in poss
    '''
    
    # Now we can sample from GEMINI at the identified locations
    j1 = model2pointsgeogcoords(simulation.xg, simulation.dat['J1'],(poss[:,0]-self.RE)*1e3,
                                    poss[:,2],90-poss[:,1])
    je = model2pointsgeogcoords(simulation.xg, simulation.dat['je'],(poss[:,0]-self.RE)*1e3,
                                    poss[:,2],90-poss[:,1])
    jn = model2pointsgeogcoords(simulation.xg, simulation.dat['jn'],(poss[:,0]-self.RE)*1e3,
                                    poss[:,2],90-poss[:,1])
    ju = model2pointsgeogcoords(simulation.xg, simulation.dat['ju'],(poss[:,0]-self.RE)*1e3,
                                    poss[:,2],90-poss[:,1])
    Be = model2pointsgeogcoords(simulation.xg, simulation.dat['Be'],(poss[:,0]-self.RE)*1e3,
                                    poss[:,2],90-poss[:,1])
    Bn = model2pointsgeogcoords(simulation.xg, simulation.dat['Bn'],(poss[:,0]-self.RE)*1e3,
                                    poss[:,2],90-poss[:,1])
    Bu = model2pointsgeogcoords(simulation.xg, simulation.dat['Bu'],(poss[:,0]-self.RE)*1e3,
                                    poss[:,2],90-poss[:,1])

    # Populate the data object
    self.lat = 90-poss[:,1]
    self.lon = poss[:,2]
    self.alt = poss[:,0]-self.RE
    self.fac = j1
    self.je = je
    self.jn = jn
    self.ju = ju
    self.Be = Be
    self.Bn = Bn
    self.Bu = Bu
    self.vperpe = model2pointsgeogcoords(simulation.xg, simulation.dat['vperpe'], 
                            (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
    self.vperpn = model2pointsgeogcoords(simulation.xg, simulation.dat['vperpn'], 
                            (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
    self.vperpu = model2pointsgeogcoords(simulation.xg, simulation.dat['vperpu'], 
                            (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
    self.jperpe = model2pointsgeogcoords(simulation.xg, simulation.dat['jperpe'], 
                            (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
    self.jperpn = model2pointsgeogcoords(simulation.xg, simulation.dat['jperpn'], 
                            (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
    self.jperpu = model2pointsgeogcoords(simulation.xg, simulation.dat['jperpu'], 
                            (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
    self.sp = model2pointsgeogcoords(simulation.xg, simulation.dat['sp'], 
                            (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
    self.sh = model2pointsgeogcoords(simulation.xg, simulation.dat['sh'], 
                            (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
    self.Ee = model2pointsgeogcoords(simulation.xg, simulation.dat['Ee'], 
                            (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
    self.En = model2pointsgeogcoords(simulation.xg, simulation.dat['En'], 
                            (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
    self.Eu = model2pointsgeogcoords(simulation.xg, simulation.dat['Eu'], 
                            (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
    self.ne = model2pointsgeogcoords(simulation.xg, simulation.dat['ne'], 
                            (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
    self.Te = model2pointsgeogcoords(simulation.xg, simulation.dat['Te'], 
                            (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
    self.Ti = model2pointsgeogcoords(simulation.xg, simulation.dat['Ti'], 
                            (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
    self.vperpmappede = model2pointsgeogcoords(simulation.xg, simulation.dat['vperpmappede'],
                            (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
    self.vperpmappedn = model2pointsgeogcoords(simulation.xg, simulation.dat['vperpmappedn'],
                            (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
    self.vperpmappedu = model2pointsgeogcoords(simulation.xg, simulation.dat['vperpmappedu'],
                            (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
    self.mappedglat = model2pointsgeogcoords(simulation.xg, simulation.dat['mappedglat'],
                            (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
    self.mappedglon = model2pointsgeogcoords(simulation.xg, simulation.dat['mappedglon'],
                            (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])              
    self.ve = model2pointsgeogcoords(simulation.xg, simulation.dat['ve'],(poss[:,0]-self.RE)*1e3,
                                    poss[:,2],90-poss[:,1])
    self.vn = model2pointsgeogcoords(simulation.xg, simulation.dat['vn'],(poss[:,0]-self.RE)*1e3,
                                    poss[:,2],90-poss[:,1])
    self.vu = model2pointsgeogcoords(simulation.xg, simulation.dat['vu'],(poss[:,0]-self.RE)*1e3,
                                    poss[:,2],90-poss[:,1])
    self.v1 = model2pointsgeogcoords(simulation.xg, simulation.dat['v1'],(poss[:,0]-self.RE)*1e3,
                                    poss[:,2],90-poss[:,1])
    self.v2 = model2pointsgeogcoords(simulation.xg, simulation.dat['v2'],(poss[:,0]-self.RE)*1e3,
                                    poss[:,2],90-poss[:,1])
    self.v3 = model2pointsgeogcoords(simulation.xg, simulation.dat['v3'],(poss[:,0]-self.RE)*1e3,
                                    poss[:,2],90-poss[:,1])

    if 'divjperp' in simulation.dat.keys():
        self.divjperp = model2pointsgeogcoords(simulation.xg, simulation.dat['divjperp'],
                                (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
    
    # Sample electric potential
    # lx1 = xg["lx"][0]
    lx2 = simulation.xg["lx"][1]
    lx3 = simulation.xg["lx"][2]
    # inds1 = range(2, lx1 + 2)
    inds2 = range(2, lx2 + 2)
    inds3 = range(2, lx3 + 2)
    # x1 = xg["x1"][inds1]
    x2 = simulation.xg["x2"][inds2]
    x3 = simulation.xg["x3"][inds3]
    x1i, x2i, x3i = geog2dipole((poss[:,0]-self.RE)*1e3, poss[:,2], 90-poss[:,1])
    xi = np.array((x2i.ravel(), x3i.ravel())).transpose()
    if len(simulation.dat.Phitop.shape) == 2:
        self.Phitop = scipy.interpolate.interpn(
            points=(x2, x3),
            values=simulation.dat.Phitop.values,
            xi=xi,
            method="linear",
            bounds_error=False,
            fill_value=np.NaN)
    if len(simulation.dat.Phitop.shape) == 3: #Not sure what this means?
        self.Phitop = scipy.interpolate.interpn(
            points=(x2, x3),
            values=simulation.dat.Phitop.values[0,:,:],
            xi=xi,
            method="linear",
            bounds_error=False,
            fill_value=np.NaN)

    # Line-of-sight sampling: NOT FULLY IMPLEMENTED YET
    # Project the mapped Vperp, at the mapped locations, onto line-of-sight 
    # direction of each measurement
    # Should try to implement an option of subtracting field aligned compoenet 
    # using a field aligned beam.
    # In reality, we will need to make such an assumption, or similar.
    # Must convert vperp into ECEF frame before doing dot product
    # enu_vec = np.vstack((selfdict['vperpmappede'],selfdict['vperpmappedn'], 
    #                      selfdict['vperpmappedu'])).T
    # xyz_vec = coordinates.enu2xyz(enu_vec, selfdict['mappedglon'], selfdict['mappedglat'])
    # selfdict['vlos'] = xyz_vec[:,0]*np.array(lx).flatten() + \
    #         xyz_vec[:,1]*np.array(ly).flatten() + \
    #         xyz_vec[:,2]*np.array(lz).flatten()
    # # convert cartesian LOS unit vector to local ENU form (at measurement location)
    # l_xyz = np.vstack((np.array(lx).flatten(),np.array(ly).flatten(),np.array(lz).flatten())).T
    # l_enu = coordinates.xyz2enu(l_xyz, selfdict['mappedglon'], selfdict['mappedglat'])
    # # Horizontal part of LOS direction
    # hormag = np.sqrt(l_enu[:,0]**2 + l_enu[:,1]**2)
    # selfdict['l_hor_e'] =l_enu[:,0]/hormag
    # selfdict['l_hor_n'] = l_enu[:,1]/hormag