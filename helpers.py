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
