#!/usr/bin/env python

''' 
   Tracking_Functions.py

   This file contains the tracking fuctions for the object
   identification and tracking of precipitation areas, cyclones,
   clouds, and moisture streams

'''

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import glob
import os
from pdb import set_trace as stop
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import median_filter
from scipy.ndimage import label
from matplotlib import cm
from scipy import ndimage
import random
import scipy
import pickle
import datetime
import pandas as pd
import subprocess
import matplotlib.path as mplPath
import sys
from calendar import monthrange
from itertools import groupby
from tqdm import tqdm
import time

#### speed up interpolation
import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
import numpy as np
import xarray as xr
import netCDF4


###########################################################
###########################################################

### UTILITY Functions
def calc_grid_distance_area(lon,lat):
    """ Function to calculate grid parameters
        It uses haversine function to approximate distances
        It approximates the first row and column to the sencond
        because coordinates of grid cell center are assumed
        lat, lon: input coordinates(degrees) 2D [y,x] dimensions
        dx: distance (m)
        dy: distance (m)
        area: area of grid cell (m2)
        grid_distance: average grid distance over the domain (m)
    """
    dy = np.zeros(lon.shape)
    dx = np.zeros(lat.shape)

    dx[:,1:]=haversine(lon[:,1:],lat[:,1:],lon[:,:-1],lat[:,:-1])
    dy[1:,:]=haversine(lon[1:,:],lat[1:,:],lon[:-1,:],lat[:-1,:])

    dx[:,0] = dx[:,1]
    dy[0,:] = dy[1,:]
    
    dx = dx * 10**3
    dy = dy * 10**3

    area = dx*dy
    grid_distance = np.mean(np.append(dy[:, :, None], dx[:, :, None], axis=2))

    return dx,dy,area,grid_distance


def radialdistance(lat1,lon1,lat2,lon2):
    # Approximate radius of earth in km
    R = 6373.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

# def haversine(lat1, lon1, lat2, lon2):

#     """Function to calculate grid distances lat-lon
#        This uses the Haversine formula
#        lat,lon : input coordinates (degrees) - array or float
#        dist_m : distance (m)
#        https://en.wikipedia.org/wiki/Haversine_formula
#        """
#     # convert decimal degrees to radians
#     lon1 = np.radians(lon1)
#     lon2 = np.radians(lon2)
#     lat1 = np.radians(lat1)
#     lat2 = np.radians(lat2)

#     # haversine formula
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#     a = np.sin(dlat / 2) ** 2 + \
#     np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
#     c = 2 * np.arcsin(np.sqrt(a))
#     # Radius of earth in kilometers is 6371
#     dist_m = c * 6371000 #const.earth_radius
#     return dist_m

def calculate_area_objects(objects_id_pr,object_indices,grid_cell_area):

    """ Calculates the area of each object during their lifetime
        one area value for each object and each timestep it exist
    """
    num_objects = len(object_indices)
    area_objects = np.array(
        [
            [
            np.sum(grid_cell_area[object_indices[obj][1:]][objects_id_pr[object_indices[obj]][tstep, :, :] == obj + 1])
            for tstep in range(objects_id_pr[object_indices[obj]].shape[0])
            ]
        for obj in range(num_objects)
        ],
    dtype=object
    )

    return area_objects

def remove_small_short_objects(objects_id,
                               area_objects,
                               min_area,
                               min_time,
                               DT,
                               objects = None):
    """Checks if the object is large enough during enough time steps
        and removes objects that do not meet this condition
        area_object: array of lists with areas of each objects during their lifetime [objects[tsteps]]
        min_area: minimum area of the object (km2)
        min_time: minimum time with the object large enough (hours)
        DT: time step of input data [hours]
        objects: object slices - speeds up processing if provided
    """

    #create final object array
    sel_objects = np.zeros(objects_id.shape,dtype=int)

    new_obj_id = 1
    for obj,_ in enumerate(area_objects):
        AreaTest = np.nanmax(
            np.convolve(
                np.array(area_objects[obj]) >= min_area * 1000**2,
                np.ones(int(min_time/ DT)),
                mode="valid",
            )
        )
        if (AreaTest == int(min_time/ DT)) & (
            len(area_objects[obj]) >= int(min_time/ DT)
        ):
            if objects == None:
                sel_objects[objects_id == (obj + 1)] =     new_obj_id
                new_obj_id += 1
            else:
                sel_objects[objects[obj]][objects_id[objects[obj]] == (obj + 1)] = new_obj_id
                new_obj_id += 1

    return sel_objects




###########################################################
###########################################################

###########################################################
###########################################################
def calc_object_characteristics(
    var_objects,  # feature object file
    var_data,  # original file used for feature detection
    filename_out,  # output file name and locaiton
    times,  # timesteps of the data
    Lat,  # 2D latidudes
    Lon,  # 2D Longitudes
    grid_spacing,       # average grid spacing
    grid_cell_area,
    min_tsteps=1,       # minimum lifetime in data timesteps
    split_merge = None  # dict containing information of splitting and merging of objects
    ):
    # ========

    num_objects = int(var_objects.max())
#     num_objects = len(np.unique(var_objects))-1
    object_indices = ndimage.find_objects(var_objects)

    if num_objects >= 1:
        objects_charac = {}
        print("            Loop over " + str(num_objects) + " objects")
        
        for iobj in range(num_objects):
            if object_indices[iobj] == None:
                continue
            object_slice = np.copy(var_objects[object_indices[iobj]])
            data_slice   = np.copy(var_data[object_indices[iobj]])

            time_idx_slice = object_indices[iobj][0]
            lat_idx_slice  = object_indices[iobj][1]
            lon_idx_slice  = object_indices[iobj][2]

            if len(object_slice) >= min_tsteps:

                data_slice[object_slice!=(iobj + 1)] = np.nan
                grid_cell_area_slice = np.tile(grid_cell_area[lat_idx_slice, lon_idx_slice], (len(data_slice), 1, 1))
                grid_cell_area_slice[object_slice != (iobj + 1)] = np.nan
                lat_slice = Lat[lat_idx_slice, lon_idx_slice]
                lon_slice = Lon[lat_idx_slice, lon_idx_slice]


                # calculate statistics
                obj_times = times[time_idx_slice]
                obj_size  = np.nansum(grid_cell_area_slice, axis=(1, 2))
                obj_min = np.nanmin(data_slice, axis=(1, 2))
                obj_max = np.nanmax(data_slice, axis=(1, 2))
                obj_mean = np.nanmean(data_slice, axis=(1, 2))
                obj_tot = np.nansum(data_slice, axis=(1, 2))


                # Track lat/lon
                obj_mass_center = \
                np.array([ndimage.measurements.center_of_mass(object_slice[tt,:,:]==(iobj+1)) for tt in range(object_slice.shape[0])])

                obj_track = np.full([len(obj_mass_center), 2], np.nan)
                iREAL = ~np.isnan(obj_mass_center[:,0])
                try:
                    obj_track[iREAL,0]=np.array([lat_slice[int(round(obj_loc[0])),int(round(obj_loc[1]))]    for tstep, obj_loc in enumerate(obj_mass_center[iREAL,:]) if np.isnan(obj_loc[0]) != True])
                    obj_track[iREAL,1]=np.array([lon_slice[int(round(obj_loc[0])),int(round(obj_loc[1]))]    for tstep, obj_loc in enumerate(obj_mass_center[iREAL,:]) if np.isnan(obj_loc[0]) != True])
                except:
                    stop()
                    
                    
#                 obj_track = np.full([len(obj_mass_center), 2], np.nan)
#                 try:
#                     obj_track[:,0]=np.array([lat_slice[int(round(obj_loc[0])),int(round(obj_loc[1]))]    for tstep, obj_loc in enumerate(obj_mass_center[:,:]) if np.isnan(obj_loc[0]) != True])
#                     obj_track[:,1]=np.array([lon_slice[int(round(obj_loc[0])),int(round(obj_loc[1]))]    for tstep, obj_loc in enumerate(obj_mass_center[:,:]) if np.isnan(obj_loc[0]) != True])
#                 except:
#                     stop()
                    
#                 if np.any(np.isnan(obj_track)):
#                     raise ValueError("track array contains NaNs")

                obj_speed = (np.sum(np.diff(obj_mass_center,axis=0)**2,axis=1)**0.5) * (grid_spacing / 1000.0)
                
                this_object_charac = {
                    "mass_center_loc": obj_mass_center,
                    "speed": obj_speed,
                    "tot": obj_tot,
                    "min": obj_min,
                    "max": obj_max,
                    "mean": obj_mean,
                    "size": obj_size,
                    #                        'rgrAccumulation':rgrAccumulation,
                    "times": obj_times,
                    "track": obj_track,
                }

                try:
                    objects_charac[str(iobj + 1)] = this_object_charac
                except:
                    raise ValueError ("Error asigning properties to final dictionary")


        if filename_out is not None:
            with open(filename_out+'.pkl', 'wb') as handle:
                pickle.dump(objects_charac, handle)

        return objects_charac

    


    
    

# This function is a predecessor of calc_object_characteristics
def ObjectCharacteristics(PR_objectsFull, # feature object file
                         PR_orig,         # original file used for feature detection
                         SaveFile,        # output file name and locaiton
                         TIME,            # timesteps of the data
                         Lat,             # 2D latidudes
                         Lon,             # 2D Longitudes
                         Gridspacing,     # average grid spacing
                         Area,
                         MinTime=1,       # minimum lifetime of an object
                         Boundary = 1):   # 1 --> remove object when it hits the boundary of the domain


    # ========

    import scipy
    import pickle

    nr_objectsUD=PR_objectsFull.max()
    rgiObjectsUDFull = PR_objectsFull
    if nr_objectsUD >= 1:
        grObject={}
        print('            Loop over '+str(PR_objectsFull.max())+' objects')
        for ob in range(int(PR_objectsFull.max())):
    #             print('        process object '+str(ob+1)+' out of '+str(PR_objectsFull.max()))
            TT=(np.sum((PR_objectsFull == (ob+1)), axis=(1,2)) > 0)
            if sum(TT) >= MinTime:
                PR_object=np.copy(PR_objectsFull[TT,:,:])
                PR_object[PR_object != (ob+1)]=0
                Objects=ndimage.find_objects(PR_object)
                if len(Objects) > 1:
                    Objects = [Objects[np.where(np.array(Objects) != None)[0][0]]]

                ObjAct = PR_object[Objects[0]]
                ValAct = PR_orig[TT,:,:][Objects[0]]
                ValAct[ObjAct == 0] = np.nan
                AreaAct = np.repeat(Area[Objects[0][1:]][None,:,:], ValAct.shape[0], axis=0)
                AreaAct[ObjAct == 0] = np.nan
                LatAct = np.copy(Lat[Objects[0][1:]])
                LonAct = np.copy(Lon[Objects[0][1:]])

                # calculate statistics
                TimeAct=TIME[TT]
                rgrSize = np.nansum(AreaAct, axis=(1,2))
                rgrPR_Min = np.nanmin(ValAct, axis=(1,2))
                rgrPR_Max = np.nanmax(ValAct, axis=(1,2))
                rgrPR_Mean = np.nanmean(ValAct, axis=(1,2))
                rgrPR_Vol = np.nansum(ValAct, axis=(1,2))

                # Track lat/lon
                rgrMassCent=np.array([scipy.ndimage.measurements.center_of_mass(ObjAct[tt,:,:]) for tt in range(ObjAct.shape[0])])
                TrackAll = np.zeros((len(rgrMassCent),2)); TrackAll[:] = np.nan
                try:
                    FIN = ~np.isnan(rgrMassCent[:,0])
                    for ii in range(len(rgrMassCent)):
                        if ~np.isnan(rgrMassCent[ii,0]) == True:
                            TrackAll[ii,1] = LatAct[int(np.round(rgrMassCent[ii][0],0)), int(np.round(rgrMassCent[ii][1],0))]
                            TrackAll[ii,0] = LonAct[int(np.round(rgrMassCent[ii][0],0)), int(np.round(rgrMassCent[ii][1],0))]
                except:
                    stop()

                rgrObjSpeed=np.array([((rgrMassCent[tt,0]-rgrMassCent[tt+1,0])**2 + (rgrMassCent[tt,1]-rgrMassCent[tt+1,1])**2)**0.5 for tt in range(ValAct.shape[0]-1)])*(Gridspacing/1000.)

                grAct={'rgrMassCent':rgrMassCent, 
                       'rgrObjSpeed':rgrObjSpeed,
                       'rgrPR_Vol':rgrPR_Vol,
                       'rgrPR_Min':rgrPR_Min,
                       'rgrPR_Max':rgrPR_Max,
                       'rgrPR_Mean':rgrPR_Mean,
                       'rgrSize':rgrSize,
    #                        'rgrAccumulation':rgrAccumulation,
                       'TimeAct':TimeAct,
                       'rgrMassCentLatLon':TrackAll}
                try:
                    grObject[str(ob+1)]=grAct
                except:
                    stop()
                    continue
        if SaveFile != None:
            pickle.dump(grObject, open(SaveFile, "wb" ) )
        return grObject
    
    
# ==============================================================
# ==============================================================

#### speed up interpolation
import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
import numpy as np
import xarray as xr

def interp_weights(xy, uv,d=2):
    tri = qhull.Delaunay(xy)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

# ==============================================================
# ==============================================================

def interpolate(values, vtx, wts):
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)


# ==============================================================
# ==============================================================
import numpy as np
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology

def detect_local_minima(arr):
    # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    # apply the local minimum filter; all locations of minimum value 
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (filters.minimum_filter(arr, footprint=neighborhood)==arr)
    # local_min is a mask that contains the peaks we are 
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    # 
    # we create the mask of the background
    background = (arr==0)
    # 
    # a little technicality: we must erode the background in order to 
    # successfully subtract it from local_min, otherwise a line will 
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    # 
    # we obtain the final mask, containing only peaks, 
    # by removing the background from the local_min mask
    detected_minima = local_min ^ eroded_background
    return np.where(detected_minima)   


# ==============================================================
# ==============================================================
def Feature_Calculation(DATA_all,    # np array that contains [time,lat,lon,Variables] with vars
                        Variables,   # Variables beeing ['V', 'U', 'T', 'Q', 'SLP']
                        dLon,        # distance between longitude cells
                        dLat,        # distance between latitude cells
                        Lat,         # Latitude coordinates
                        dT,          # time step in hours
                        Gridspacing):# grid spacing in m
    from scipy import ndimage
    
    
    # 11111111111111111111111111111111111111111111111111
    # calculate vapor transport on pressure level
    VapTrans = ((DATA_all[:,:,:,Variables.index('U')]*DATA_all[:,:,:,Variables.index('Q')])**2 + (DATA_all[:,:,:,Variables.index('V')]*DATA_all[:,:,:,Variables.index('Q')])**2)**(1/2)

    # 22222222222222222222222222222222222222222222222222
    # Frontal Detection according to https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017GL073662
    UU = DATA_all[:,:,:,Variables.index('U')]
    VV = DATA_all[:,:,:,Variables.index('V')]
    dx = dLon
    dy = dLat
    du = np.gradient( UU )
    dv = np.gradient( VV )
    PV = np.abs( dv[-1]/dx[None,:] - du[-2]/dy[None,:] )
    TK = DATA_all[:,:,:,Variables.index('T')]
    vgrad = np.gradient(TK, axis=(1,2))
    Tgrad = np.sqrt(vgrad[0]**2 + vgrad[1]**2)

    Fstar = PV * Tgrad

    Tgrad_zero = 0.45#*100/(np.mean([dLon,dLat], axis=0)/1000.)  # 0.45 K/(100 km)
    import metpy.calc as calc
    from metpy.units import units
    CoriolisPar = calc.coriolis_parameter(np.deg2rad(Lat))
    Frontal_Diagnostic = np.array(Fstar/(CoriolisPar * Tgrad_zero))

    # # 3333333333333333333333333333333333333333333333333333
    # # Cyclone identification based on pressure annomaly threshold

    SLP = DATA_all[:,:,:,Variables.index('SLP')]/100.
    # remove high-frequency variabilities --> smooth over 100 x 100 km (no temporal smoothing)
    SLP_smooth = ndimage.uniform_filter(SLP, size=[1,int(100/(Gridspacing/1000.)),int(100/(Gridspacing/1000.))])
    # smoothign over 3000 x 3000 km and 78 hours
    SLPsmoothAn = ndimage.uniform_filter(SLP, size=[int(78/dT),int(int(3000/(Gridspacing/1000.))),int(int(3000/(Gridspacing/1000.)))])
    SLP_Anomaly = np.array(SLP_smooth-SLPsmoothAn)
    # plt.contour(SLP_Anomaly[tt,:,:], levels=[-9990,-10,1100], colors='b')
    Pressure_anomaly = SLP_Anomaly < -12 # 12 hPa depression
    HighPressure_annomaly = SLP_Anomaly > 12

    return Pressure_anomaly, Frontal_Diagnostic, VapTrans, SLP_Anomaly, vgrad, HighPressure_annomaly



# ==============================================================
# ==============================================================
# from math import radians, cos, sin, asin, sqrt
# def haversine(lon1, lat1, lon2, lat2):
#     """
#     Calculate the great circle distance between two points 
#     on the earth (specified in decimal degrees)
#     """
#     # convert decimal degrees to radians 
#     lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
#     # haversine formula 
#     dlon = lon2 - lon1 
#     dlat = lat2 - lat1 
#     a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
#     c = 2 * asin(sqrt(a)) 
#     # Radius of earth in kilometers is 6371
#     km = 6371* c
#     return km



def ReadERA5(TIME,      # Time period to read (this program will read hourly data)
            var,        # Variable name. See list below for defined variables
            PL,         # Pressure level of variable
            REGION):    # Region to read. Format must be <[N,E,S,W]> in degrees from -180 to +180 longitude
    # ----------
    # This function reads hourly ERA5 data for one variable from NCAR's RDA archive in a region of interest.
    # ----------

    DayStart = datetime.datetime(TIME[0].year, TIME[0].month, TIME[0].day,TIME[0].hour)
    DayStop = datetime.datetime(TIME[-1].year, TIME[-1].month, TIME[-1].day,TIME[-1].hour)
    TimeDD=pd.date_range(DayStart, end=DayStop, freq='d')
    Plevels = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000])

    dT = int(divmod((TimeDD[1] - TimeDD[0]).total_seconds(), 60)[0]/60)
    
    # check if variable is defined
    if var == 'V':
        ERAvarfile = 'v.ll025uv'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.pl/'
        NCvarname = 'V'
        PL = np.argmin(np.abs(Plevels - PL))
    if var == 'U':
        ERAvarfile = 'u.ll025uv'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.pl/'
        NCvarname = 'U'
        PL = np.argmin(np.abs(Plevels - PL))
    if var == 'T':
        ERAvarfile = 't.ll025sc'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.pl/'
        NCvarname = 'T'
        PL = np.argmin(np.abs(Plevels - PL))
    if var == 'ZG':
        ERAvarfile = 'z.ll025sc'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.pl/'
        NCvarname = 'Z'
        PL = np.argmin(np.abs(Plevels - PL))
    if var == 'Q':
        ERAvarfile = 'q.ll025sc'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.pl/'
        NCvarname = 'Q'
        PL = np.argmin(np.abs(Plevels - PL))
    if var == 'SLP':
        ERAvarfile = 'msl.ll025sc'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.sfc/'
        NCvarname = 'MSL'
        PL = -1
    if var == 'IVTE':
        ERAvarfile = 'viwve.ll025sc'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.vinteg/'
        NCvarname = 'VIWVE'
        PL = -1
    if var == 'IVTN':
        ERAvarfile = 'viwvn.ll025sc'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.vinteg/'
        NCvarname = 'VIWVN'
        PL = -1

    print(ERAvarfile)
    # read in the coordinates
    ncid=Dataset("/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.invariant/197901/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc", mode='r')
    Lat=np.squeeze(ncid.variables['latitude'][:])
    Lon=np.squeeze(ncid.variables['longitude'][:])
    # Zfull=np.squeeze(ncid.variables['Z'][:])
    ncid.close()
    if np.max(Lon) > 180:
        Lon[Lon >= 180] = Lon[Lon >= 180] - 360
    Lon,Lat = np.meshgrid(Lon,Lat)

    # get the region of interest
    if (REGION[1] > 0) & (REGION[3] < 0):
        # region crosses zero meridian
        iRoll = np.sum(Lon[0,:] < 0)
    else:
        iRoll=0
    Lon = np.roll(Lon,iRoll, axis=1)
    iNorth = np.argmin(np.abs(Lat[:,0] - REGION[0]))
    iSouth = np.argmin(np.abs(Lat[:,0] - REGION[2]))+1
    iEeast = np.argmin(np.abs(Lon[0,:] - REGION[1]))+1
    iWest = np.argmin(np.abs(Lon[0,:] - REGION[3]))
    print(iNorth,iSouth,iWest,iEeast)

    Lon = Lon[iNorth:iSouth,iWest:iEeast]
    Lat = Lat[iNorth:iSouth,iWest:iEeast]
    # Z=np.roll(Zfull,iRoll, axis=1)
    # Z = Z[iNorth:iSouth,iWest:iEeast]

    DataAll = np.zeros((len(TIME),Lon.shape[0],Lon.shape[1])); DataAll[:]=np.nan
    tt=0
    
    for mm in range(len(TimeDD)):
        YYYYMM = str(TimeDD[mm].year)+str(TimeDD[mm].month).zfill(2)
        YYYYMMDD = str(TimeDD[mm].year)+str(TimeDD[mm].month).zfill(2)+str(TimeDD[mm].day).zfill(2)
        DirAct = Dir + YYYYMM + '/'
        if (var == 'SLP') | (var == 'IVTE') | (var == 'IVTN'):
            FILES = glob.glob(DirAct + '*'+ERAvarfile+'*'+YYYYMM+'*.nc')
        else:
            FILES = glob.glob(DirAct + '*'+ERAvarfile+'*'+YYYYMMDD+'*.nc')
        FILES = np.sort(FILES)
        
        TIMEACT = TIME[(TimeDD[mm].year == TIME.year) &  (TimeDD[mm].month == TIME.month) & (TimeDD[mm].day == TIME.day)]
        
        for fi in range(len(FILES)): #[7:9]:
            print(FILES[fi])
            ncid = Dataset(FILES[fi], mode='r')
            time_var = ncid.variables['time']
            dtime = netCDF4.num2date(time_var[:],time_var.units)
            TimeNC = pd.to_datetime([pd.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in dtime])
            TT = np.isin(TimeNC, TIMEACT)
            if iRoll != 0:
                if PL !=-1:
                    try:
                        DATAact = np.squeeze(ncid.variables[NCvarname][TT,PL,iNorth:iSouth,:])
                    except:
                        stop()
                else:
                    DATAact = np.squeeze(ncid.variables[NCvarname][TT,iNorth:iSouth,:])
                ncid.close()
            else:
                if PL !=-1:
                    DATAact = np.squeeze(ncid.variables[NCvarname][TT,PL,iNorth:iSouth,iWest:iEeast])
                else:
                    DATAact = np.squeeze(ncid.variables[NCvarname][TT,iNorth:iSouth,iWest:iEeast])
                ncid.close()
            # cut out region
            if len(DATAact.shape) == 2:
                DATAact=DATAact[None,:,:]
            DATAact=np.roll(DATAact,iRoll, axis=2)
            if iRoll != 0:
                DATAact = DATAact[:,:,iWest:iEeast]
            else:
                DATAact = DATAact[:,:,:]
            try:
                DataAll[tt:tt+DATAact.shape[0],:,:]=DATAact
            except:
                continue
            tt = tt+DATAact.shape[0]
    return DataAll, Lat, Lon



# =======================================================================================
def ReadERA5_2D(TIME,      # Time period to read (this program will read hourly data)
            var,        # Variable name. See list below for defined variables
            PL,         # Pressure level of variable
            REGION):    # Region to read. Format must be <[N,E,S,W]> in degrees from -180 to +180 longitude
    # ----------
    # This function reads hourly 2D ERA5 data.
    # ----------
    from calendar import monthrange

    DayStart = datetime.datetime(TIME[0].year, TIME[0].month, TIME[0].day,TIME[0].hour)
    DayStop = datetime.datetime(TIME[-1].year, TIME[-1].month, TIME[-1].day,TIME[-1].hour)
    TimeDD=pd.date_range(DayStart, end=DayStop, freq='d')
    TimeMM=pd.date_range(DayStart, end=DayStop, freq='m')
    if len(TimeMM) == 0:
        TimeMM = [TimeDD[0]]

    dT = int(divmod((TimeDD[1] - TimeDD[0]).total_seconds(), 60)[0]/60)
    ERA5dir = '/glade/campaign/mmm/c3we/prein/ERA5/hourly/'
    if PL != -1:
        DirName = str(var)+str(PL)
    else:
        DirName = str(var)

    print(var)
    # read in the coordinates
    ncid=Dataset("/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.invariant/197901/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc", mode='r')
    Lat=np.squeeze(ncid.variables['latitude'][:])
    Lon=np.squeeze(ncid.variables['longitude'][:])
    # Zfull=np.squeeze(ncid.variables['Z'][:])
    ncid.close()
    if np.max(Lon) > 180:
        Lon[Lon >= 180] = Lon[Lon >= 180] - 360
    Lon,Lat = np.meshgrid(Lon,Lat)

    # get the region of interest
    if (REGION[1] > 0) & (REGION[3] < 0):
        # region crosses zero meridian
        iRoll = np.sum(Lon[0,:] < 0)
    else:
        iRoll=0
    Lon = np.roll(Lon,iRoll, axis=1)
    iNorth = np.argmin(np.abs(Lat[:,0] - REGION[0]))
    iSouth = np.argmin(np.abs(Lat[:,0] - REGION[2]))+1
    iEeast = np.argmin(np.abs(Lon[0,:] - REGION[1]))+1
    iWest = np.argmin(np.abs(Lon[0,:] - REGION[3]))
    print(iNorth,iSouth,iWest,iEeast)

    Lon = Lon[iNorth:iSouth,iWest:iEeast]
    Lat = Lat[iNorth:iSouth,iWest:iEeast]
    # Z=np.roll(Zfull,iRoll, axis=1)
    # Z = Z[iNorth:iSouth,iWest:iEeast]

    DataAll = np.zeros((len(TIME),Lon.shape[0],Lon.shape[1])); DataAll[:]=np.nan
    tt=0
    
    for mm in tqdm(range(len(TimeMM))):
        YYYYMM = str(TimeMM[mm].year)+str(TimeMM[mm].month).zfill(2)
        YYYY = TimeMM[mm].year
        MM = TimeMM[mm].month
        DD = monthrange(YYYY, MM)[1]
        
        TimeFile = TimeDD=pd.date_range(datetime.datetime(YYYY, MM, 1,0), end=datetime.datetime(YYYY, MM, DD,23), freq='h')
        TT = np.isin(TimeFile,TIME)
        
        ncid = Dataset(ERA5dir+DirName+'/'+YYYYMM+'_'+DirName+'_ERA5.nc', mode='r')
        if iRoll != 0:
            DATAact = np.squeeze(ncid.variables[var][TT,iNorth:iSouth,:])
            ncid.close()
        else:
            DATAact = np.squeeze(ncid.variables[var][TT,iNorth:iSouth,iWest:iEeast])
            ncid.close()
        # cut out region
        if len(DATAact.shape) == 2:
            DATAact=DATAact[None,:,:]
        DATAact=np.roll(DATAact,iRoll, axis=2)
        if iRoll != 0:
            DATAact = DATAact[:,:,iWest:iEeast]
        try:
            DataAll[tt:tt+DATAact.shape[0],:,:]=DATAact
        except:
            continue
        tt = tt+DATAact.shape[0]
    return DataAll, Lat, Lon

def ConnectLon(object_indices):
    for tt in range(object_indices.shape[0]):
        EDGE = np.append(
            object_indices[tt, :, -1][:, None], object_indices[tt, :, 0][:, None], axis=1
        )
        iEDGE = np.sum(EDGE > 0, axis=1) == 2
        OBJ_Left = EDGE[iEDGE, 0]
        OBJ_Right = EDGE[iEDGE, 1]
        OBJ_joint = np.array(
            [
                OBJ_Left[ii].astype(str) + "_" + OBJ_Right[ii].astype(str)
                for ii,_ in enumerate(OBJ_Left)
            ]
        )
        NotSame = OBJ_Left != OBJ_Right
        OBJ_joint = OBJ_joint[NotSame]
        OBJ_unique = np.unique(OBJ_joint)
        # set the eastern object to the number of the western object in all timesteps
        for obj,_ in enumerate(OBJ_unique):
            ObE = int(OBJ_unique[obj].split("_")[1])
            ObW = int(OBJ_unique[obj].split("_")[0])
            object_indices[object_indices == ObE] = ObW
    return object_indices


def ConnectLon_on_timestep(object_indices):
    
    """ This function connects objects over the date line on a time-step by
        time-step basis, which makes it different from the ConnectLon function.
        This function is needed when long-living objects are first split into
        smaller objects using the BreakupObjects function.
    """
    
    for tt in range(object_indices.shape[0]):
        EDGE = np.append(
            object_indices[tt, :, -1][:, None], object_indices[tt, :, 0][:, None], axis=1
        )
        iEDGE = np.sum(EDGE > 0, axis=1) == 2
        OBJ_Left = EDGE[iEDGE, 0]
        OBJ_Right = EDGE[iEDGE, 1]
        OBJ_joint = np.array(
            [
                OBJ_Left[ii].astype(str) + "_" + OBJ_Right[ii].astype(str)
                for ii,_ in enumerate(OBJ_Left)
            ]
        )
        NotSame = OBJ_Left != OBJ_Right
        OBJ_joint = OBJ_joint[NotSame]
        OBJ_unique = np.unique(OBJ_joint)
        # set the eastern object to the number of the western object in all timesteps
        for obj,_ in enumerate(OBJ_unique):
            ObE = int(OBJ_unique[obj].split("_")[1])
            ObW = int(OBJ_unique[obj].split("_")[0])
            object_indices[tt,object_indices[tt,:] == ObE] = ObW
    return object_indices


### Break up long living objects by extracting the biggest object at each time
def BreakupObjects(
    DATA,  # 3D matrix [time,lat,lon] containing the objects
    min_tsteps,  # minimum lifetime in data timesteps
    dT,# time step in hours
    obj_history = False,  # calculates how object start and end
    ):  

    start = time.perf_counter()

    object_indices = ndimage.find_objects(DATA)
    MaxOb = np.max(DATA)
    MinLif = int(min_tsteps / dT)  # min lifetime of object to be split
    AVmax = 1.5

    obj_structure_2D = np.zeros((3, 3, 3))
    obj_structure_2D[1, :, :] = 1
    rgiObjects2D, nr_objects2D = ndimage.label(DATA, structure=obj_structure_2D)

    rgiObjNrs = np.unique(DATA)[1:]
    TT = np.zeros((MaxOb))
    for obj in range(MaxOb):  
        if object_indices[obj] != None:
            TT[obj] = object_indices[obj][0].stop - object_indices[obj][0].start
    TT = TT[rgiObjNrs-1]
    TT = TT.astype('int')
    # Sel_Obj = rgiObjNrs[TT > MinLif]

    # Average 2D objects in 3D objects?
    Av_2Dob = np.zeros((len(rgiObjNrs)))
    Av_2Dob[:] = np.nan
    ii = 1
    
    object_split = {} # this directory holds information about splitting and merging of objects
    for obj in tqdm(range(len(rgiObjNrs))):
        iOb = rgiObjNrs[obj]
        if TT[obj] <= MinLif:
            # ignore short lived objects
            DATA[DATA == iOb] = 0
            continue
        SelOb = rgiObjNrs[obj] - 1
        DATA_ACT = np.copy(DATA[object_indices[SelOb]])
        rgiObjects2D_ACT = np.copy(rgiObjects2D[object_indices[SelOb]])
        rgiObjects2D_ACT[DATA_ACT != iOb] = 0

        Av_2Dob[obj] = np.mean(
            np.array(
                [
                    len(np.unique(rgiObjects2D_ACT[tt, :, :])) - 1
                    for tt in range(DATA_ACT.shape[0])
                ]
            )
        )

        if Av_2Dob[obj] <= AVmax:
            if obj_history == True:
                # this is a signle[ object
                object_split[str(iOb)] = [0] * TT[obj]
                if object_indices[SelOb][0].start == 0:
                    # object starts when tracking starts
                    object_split[str(iOb)][0] = -1
                if object_indices[SelOb][0].stop == DATA.shape[0]-1:
                    # object stops when tracking stops
                    object_split[str(iOb)][-1] = -1
        else:
            rgiObAct = np.unique(rgiObjects2D_ACT[0, :, :])[1:]
            for tt in range(1, rgiObjects2D_ACT[:, :, :].shape[0]):
                rgiObActCP = list(np.copy(rgiObAct))
                for ob1 in rgiObAct:
                    tt1_obj = list(
                        np.unique(
                            rgiObjects2D_ACT[tt, rgiObjects2D_ACT[tt - 1, :] == ob1]
                        )[1:]
                    )
                    if len(tt1_obj) == 0:
                        # this object ends here
                        rgiObActCP.remove(ob1)
                        continue
                    elif len(tt1_obj) == 1:
                        rgiObjects2D_ACT[
                            tt, rgiObjects2D_ACT[tt, :] == tt1_obj[0]
                        ] = ob1
                    else:
                        VOL = [
                            np.sum(rgiObjects2D_ACT[tt, :] == tt1_obj[jj])
                            for jj,_ in enumerate(tt1_obj)
                        ]
                        rgiObjects2D_ACT[
                            tt, rgiObjects2D_ACT[tt, :] == tt1_obj[np.argmax(VOL)]
                        ] = ob1
                        tt1_obj.remove(tt1_obj[np.argmax(VOL)])
                        rgiObActCP = rgiObActCP + list(tt1_obj)

                # make sure that mergers are assigned the largest object
                for ob2 in rgiObActCP:
                    ttm1_obj = list(
                        np.unique(
                            rgiObjects2D_ACT[tt - 1, rgiObjects2D_ACT[tt, :] == ob2]
                        )[1:]
                    )
                    if len(ttm1_obj) > 1:
                        VOL = [
                            np.sum(rgiObjects2D_ACT[tt - 1, :] == ttm1_obj[jj])
                            for jj,_ in enumerate(ttm1_obj)
                        ]
                        rgiObjects2D_ACT[tt, rgiObjects2D_ACT[tt, :] == ob2] = ttm1_obj[
                            np.argmax(VOL)
                        ]

                # are there new object?
                NewObj = np.unique(rgiObjects2D_ACT[tt, :, :])[1:]
                NewObj = list(np.setdiff1d(NewObj, rgiObAct))
                if len(NewObj) != 0:
                    rgiObActCP = rgiObActCP + NewObj
                rgiObActCP = np.unique(rgiObActCP)
                rgiObAct = np.copy(rgiObActCP)

            rgiObjects2D_ACT[rgiObjects2D_ACT != 0] = np.copy(
                rgiObjects2D_ACT[rgiObjects2D_ACT != 0] + MaxOb
            )
            MaxOb = np.max(DATA)

            # save the new objects to the original object array
            TMP = np.copy(DATA[object_indices[SelOb]])
            TMP[rgiObjects2D_ACT != 0] = rgiObjects2D_ACT[rgiObjects2D_ACT != 0]
            DATA[object_indices[SelOb]] = np.copy(TMP)

            if obj_history == True:
                # ----------------------------------
                # remember how objects start and end
                temp_obj = np.unique(TMP[DATA_ACT[:, :, :] == iOb])
                for ob_ms in range(len(temp_obj)):
                    t1_obj = temp_obj[ob_ms]
                    sel_time = np.where(np.sum((TMP == t1_obj) > 0, axis=(1,2)) > 0)[0]
                    obj_charac = [0] * len(sel_time)
                    for kk in range(len(sel_time)):
                        if sel_time[kk] == 0:
                            # object starts when tracking starts
                            obj_charac[kk] = -1
                        elif sel_time[kk]+1 == TMP.shape[0]:
                            # object ends when tracking ends
                            obj_charac[kk] = -1

                        # check if system starts from splitting
                        t0_ob = TMP[sel_time[kk]-1,:,:][TMP[sel_time[kk],:,:] == t1_obj]
                        unique_t0 = list(np.unique(t0_ob))
                        try:
                            unique_t0.remove(0)
                        except:
                            pass
                        try:
                            unique_t0.remove(t1_obj)
                        except:
                            pass
                        if len(unique_t0) == 0:
                            # object has pure start or continues without interactions
                            continue
                        else:
                            # Object merges with other object
                            obj_charac[kk] = unique_t0[0]

                    # check if object ends by merging
                    if obj_charac[-1] != -1:
                        if sel_time[-1]+1 == TMP.shape[0]:
                            obj_charac[-1] = -1
                        else:
                            t2_ob = TMP[sel_time[-1]+1,:,:][TMP[sel_time[-1],:,:] == t1_obj]
                            unique_t2 = list(np.unique(t2_ob))
                            try:
                                unique_t2.remove(0)
                            except:
                                pass
                            try:
                                unique_t2.remove(t1_obj)
                            except:
                                pass
                            if len(unique_t2) != 0:
                                obj_charac[-1] = unique_t2[0]

                    object_split[str(t1_obj)] = obj_charac

    # clean up object matrix
    if obj_history == True:
        DATA_fin, object_split =    clean_up_objects(DATA,
                                    dT,
                                    min_tsteps, 
                                    obj_splitmerge = object_split)
    else:
        DATA_fin, object_split =    clean_up_objects(DATA,
                                    dT,
                                    min_tsteps)

    end = time.perf_counter()
    timer(start, end)

    return DATA_fin, object_split




# from https://stackoverflow.com/questions/27779677/how-to-format-elapsed-time-from-seconds-to-hours-minutes-seconds-and-milliseco
def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("        "+"{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


# this function removes nan values by interpolating temporally
from numba import jit
@jit(nopython=True)
def interpolate_numba(arr, no_data=-32768):
    """return array interpolated along time-axis to fill missing values"""
    result = np.zeros_like(arr, dtype=np.int16)

    for x in range(arr.shape[2]):
        # slice along x axis
        for y in range(arr.shape[1]):
            # slice along y axis
            for z in range(arr.shape[0]):
                value = arr[z,y,x]
                if z == 0:  # don't interpolate first value
                    new_value = value
                elif z == len(arr[:,0,0])-1:  # don't interpolate last value
                    new_value = value

                elif value == no_data:  # interpolate

                    left = arr[z-1,y,x]
                    right = arr[z+1,y,x]
                    # look for valid neighbours
                    if left != no_data and right != no_data:  # left and right are valid
                        new_value = (left + right) / 2

                    elif left == no_data and z == 1:  # boundary condition left
                        new_value = value
                    elif right == no_data and z == len(arr[:,0,0])-2:  # boundary condition right
                        new_value = value

                    elif left == no_data and right != no_data:  # take second neighbour to the left
                        more_left = arr[z-2,y,x]
                        if more_left == no_data:
                            new_value = value
                        else:
                            new_value = (more_left + right) / 2

                    elif left != no_data and right == no_data:  # take second neighbour to the right
                        more_right = arr[z+2,y,x]
                        if more_right == no_data:
                            new_value = value
                        else:
                            new_value = (more_right + left) / 2

                    elif left == no_data and right == no_data:  # take second neighbour on both sides
                        more_left = arr[z-2,y,x]
                        more_right = arr[z+2,y,x]
                        if more_left != no_data and more_right != no_data:
                            new_value = (more_left + more_right) / 2
                        else:
                            new_value = value
                    else:
                        new_value = value
                else:
                    new_value = value
                result[z,y,x] = int(new_value)
    return result


# ============================================================
#              tropical wave classification
# https://github.com/tmiyachi/mcclimate/blob/master/kf_filter.py

import numpy
import scipy.fftpack as fftpack
import scipy.signal as signal
import sys

gravitational_constant = 6.673e-11 #Nm^2/kg^2
gasconst = 8.314e+3 #JK^-1kmol^-1

#earth
gravity_earth = 9.81 #m/s^2
radius_earth = 6.37e+6
omega_earth = 7.292e-5

#air
gasconst_dry = 287
specific_heat_pressure = 1004
specific_heat_volume = 717
ratio_gamma = specific_heat_pressure/specific_heat_volume

NA = numpy.newaxis
pi = numpy.pi
g = gravity_earth
a = radius_earth
beta = 2.0*omega_earth/radius_earth


class KFfilter:
    """class for wavenumber-frequency filtering for WK99 and WKH00"""
    def __init__(self, datain, spd, tim_taper=0.1):
        """Arguments:
        
       'datain'    -- the data to be filtered. dimension must be (time, lat, lon)

       'spd'       -- samples per day

       'tim_taper' -- tapering ratio by cos. applay tapering first and last tim_taper%
                      samples. default is cos20 tapering

                      """
        ntim, nlat, nlon = datain.shape

        #remove the lowest three harmonics of the seasonal cycle (WK99, WKW03)
##         if ntim > 365*spd/3:
##             rf = fftpack.rfft(datain,axis=0)
##             freq = fftpack.rfftfreq(ntim*spd, d=1./float(spd))
##             rf[(freq <= 3./365) & (freq >=1./365),:,:] = 0.0     #freq<=3./365 only??
##             datain = fftpack.irfft(rf,axis=0)

        #remove dominal trend
        data = signal.detrend(datain, axis=0)

        #tapering
        if tim_taper == 'hann':
            window = signal.hann(ntim)
            data = data * window[:,NA,NA]
        elif tim_taper > 0:
        #taper by cos tapering same dtype as input array
            tp = int(ntim*tim_taper)
            window = numpy.ones(ntim, dtype=datain.dtype)
            x = numpy.arange(tp)
            window[:tp] = 0.5*(1.0-numpy.cos(x*pi/tp))
            window[-tp:] = 0.5*(1.0-numpy.cos(x[::-1]*pi/tp))
            data = data * window[:,NA,NA]

        #FFT
        self.fftdata = fftpack.fft2(data, axes=(0,2))

        #Note
        # fft is defined by exp(-ikx), so to adjust exp(ikx) multipried minus         
        wavenumber = -fftpack.fftfreq(nlon)*nlon
        frequency = fftpack.fftfreq(ntim, d=1./float(spd))
        knum, freq = numpy.meshgrid(wavenumber, frequency)

        #make f<0 domain same as f>0 domain
        #CAUTION: wave definition is exp(i(k*x-omega*t)) but FFT definition exp(-ikx)
        #so cahnge sign
        knum[freq<0] = -knum[freq<0]
        freq = numpy.abs(freq)
        self.knum = knum
        self.freq = freq

        self.wavenumber = wavenumber
        self.frequency = frequency

    def decompose_antisymm(self):
        """decompose attribute data to sym and antisym component
        """
        fftdata = self.fftdata
        nf, nlat, nk = fftdata.shape
        symm = 0.5*(fftdata[:,:nlat/2+1,:] + fftdata[:,nlat:nlat/2-1:-1,:])  
        anti = 0.5*(fftdata[:,:nlat/2,:] - fftdata[:,nlat:nlat/2:-1,:]) 
        
        self.fftdata = numpy.concatenate([anti, symm],axis=1)

    def kfmask(self, fmin=None, fmax=None, kmin=None, kmax=None):
        """return wavenumber-frequency mask for wavefilter method

        Arguments:

           'fmin/fmax' --

           'kmin/kmax' --
        """
        nf, nlat, nk = self.fftdata.shape
        knum = self.knum
        freq = self.freq

        #wavenumber cut-off
        mask = numpy.zeros((nf,nk), dtype=numpy.bool)
        if kmin != None:
            mask = mask | (knum < kmin) 
        if kmax != None:
            mask = mask | (kmax < knum)

        #frequency cutoff
        if fmin != None:
            mask = mask | (freq < fmin)
        if fmax != None:
            mask = mask | (fmax < freq)

        return mask

    def wavefilter(self, mask):
        """apply wavenumber-frequency filtering by original mask.
        
        Arguments:
        
           'mask' -- 2D boolean array (wavenumber, frequency).domain to be filterd
                     is False (True member to be zero)
        """
        wavenumber = self.wavenumber
        frequency = self.frequency
        fftdata = self.fftdata.copy()
        nf, nlat, nk = fftdata.shape

        if (nf, nk) != mask.shape:
            print( "mask array size is incorrect.")
            sys.exit()

        mask = numpy.repeat(mask[:,NA,:], nlat, axis=1)    
        fftdata[mask] = 0.0

        #inverse FFT
        filterd = fftpack.ifft2(fftdata, axes=(0,2))
        return filterd.real

    #filter
    def kelvinfilter(self, fmin=0.05, fmax=0.4, kmin=None, kmax=14, hmin=8, hmax=90):
        """kelvin wave filter

        Arguments:

           'fmin/fmax' -- unit is cycle per day

           'kmin/kmax' -- zonal wave number

           'hmin/hmax' --equivalent depth
        """
        
        fftdata = self.fftdata.copy()
        knum = self.knum
        freq = self.freq
        nf, nlat, nk = fftdata.shape

        # filtering ############################################################
        mask = numpy.zeros((nf,nk), dtype=numpy.bool)
        #wavenumber cut-off
        if kmin != None:
            mask = mask | (knum < kmin) 
        if kmax != None:
            mask = mask | (kmax < knum)

        #frequency cutoff
        if fmin != None:
            mask = mask | (freq < fmin)
        if fmax != None:
            mask = mask | (fmax < freq)

        #dispersion filter
        if hmin != None:
            c = numpy.sqrt(g*hmin)
            omega = 2.*pi*freq/24./3600. / numpy.sqrt(beta*c) #adusting day^-1 to s^-1
            k     = knum/a * numpy.sqrt(c/beta)         #adusting ^2pia to ^m
            mask = mask | (omega - k <0)
        if hmax != None:
            c = numpy.sqrt(g*hmax)
            omega = 2.*pi*freq/24./3600. / numpy.sqrt(beta*c) #adusting day^-1 to s^-1
            k     = knum/a * numpy.sqrt(c/beta)         #adusting ^2pia to ^m
            mask = mask | (omega - k >0)

        mask = numpy.repeat(mask[:,NA,:], nlat, axis=1)
        fftdata[mask] = 0.0

        filterd = fftpack.ifft2(fftdata, axes=(0,2))
        return filterd.real

    def erfilter(self, fmin=None, fmax=None, kmin=-10, kmax=-1, hmin=8, hmax=90, n=1):
        """equatorial wave filter

        Arguments:

           'fmin/fmax' -- unit is cycle per day

           'kmin/kmax' -- zonal wave number

           'hmin/hmax' -- equivalent depth

           'n'         -- meridional mode number
        """

        if n <=0 or n%1 !=0:
            print("n must be n>=1 integer")
            sys.exit()

        fftdata = self.fftdata.copy()
        knum = self.knum
        freq = self.freq
        nf, nlat, nk = fftdata.shape

        # filtering ############################################################
        mask = numpy.zeros((nf,nk), dtype=numpy.bool)
        #wavenumber cut-off
        if kmin != None:
            mask = mask | (knum < kmin) 
        if kmax != None:
            mask = mask | (kmax < knum)

        #frequency cutoff
        if fmin != None:
            mask = mask | (freq < fmin)
        if fmax != None:
            mask = mask | (fmax < freq)

        #dispersion filter
        if hmin != None:
            c = numpy.sqrt(g*hmin)
            omega = 2.*pi*freq/24./3600. / numpy.sqrt(beta*c)           #adusting day^-1 to s^-1
            k     = knum/a * numpy.sqrt(c/beta)                         #adusting ^2pia to ^m               
            mask = mask | (omega*(k**2 + (2*n+1)) + k  < 0)
        if hmax != None:
            c = numpy.sqrt(g*hmax)
            omega = 2.*pi*freq/24./3600. / numpy.sqrt(beta*c)           #adusting day^-1 to s^-1
            k     = knum/a * numpy.sqrt(c/beta)                         #adusting ^2pia to ^m               
            mask = mask | (omega*(k**2 + (2*n+1)) + k  > 0)
        mask = numpy.repeat(mask[:,NA,:], nlat, axis=1)
        
        fftdata[mask] = 0.0

        filterd = fftpack.ifft2(fftdata, axes=(0,2))
        return filterd.real

    def igfilter(self, fmin=None, fmax=None, kmin=-15, kmax=-1, hmin=12, hmax=90, n=1):
        """n>=1 inertio gravirt wave filter. default is n=1 WIG.

        Arguments:

           'fmin/fmax' -- unit is cycle per day

           'kmin/kmax' -- zonal wave number. negative is westward, positive is
                          eastward

           'hmin/hmax' -- equivalent depth

           'n'         -- meridional mode number
        """
        if n <=0 or n%1 !=0:
            print("n must be n>=1 integer. for n=0 EIG you must use eig0filter method.")
            sys.exit()

        fftdata = self.fftdata.copy()
        knum = self.knum
        freq = self.freq
        nf, nlat, nk = fftdata.shape

        # filtering ############################################################
        mask = numpy.zeros((nf,nk), dtype=numpy.bool)
        #wavenumber cut-off
        if kmin != None:
            mask = mask | (knum < kmin) 
        if kmax != None:
            mask = mask | (kmax < knum)

        #frequency cutoff
        if fmin != None:
            mask = mask | (freq < fmin)
        if fmax != None:
            mask = mask | (fmax < freq)

        #dispersion filter
        if hmin != None:
            c = numpy.sqrt(g*hmin)
            omega = 2.*pi*freq/24./3600. / numpy.sqrt(beta*c)           #adusting day^-1 to s^-1
            k     = knum/a * numpy.sqrt(c/beta)                         #adusting ^2pia to ^m               
            mask = mask | (omega**2 - k**2 - (2*n+1)  < 0)
        if hmax != None:
            c = numpy.sqrt(g*hmax)
            omega = 2.*pi*freq/24./3600. / numpy.sqrt(beta*c)           #adusting day^-1 to s^-1
            k     = knum/a * numpy.sqrt(c/beta)                         #adusting ^2pia to ^m               
            mask = mask | (omega**2 - k**2 - (2*n+1)  > 0)
        mask = numpy.repeat(mask[:,NA,:], nlat, axis=1)
        fftdata[mask] = 0.0
        
        filterd = fftpack.ifft2(fftdata, axes=(0,2))
        return filterd.real

    def eig0filter(self, fmin=None, fmax=0.55, kmin=0, kmax=15, hmin=12, hmax=50):
        """n>=0 eastward inertio gravirt wave filter.

        Arguments:

           'fmin/fmax' -- unit is cycle per day

           'kmin/kmax' -- zonal wave number. negative is westward, positive is
                          eastward

           'hmin/hmax' -- equivalent depth
        """
        if kmin < 0:
            print("kmin must be positive. if k < 0, this mode is MRG")
            sys.exit()

        fftdata = self.fftdata.copy()
        knum = self.knum
        freq = self.freq
        nf, nlat, nk = fftdata.shape

        # filtering ############################################################
        mask = numpy.zeros((nf,nk), dtype=numpy.bool)
        #wavenumber cut-off
        if kmin != None:
            mask = mask | (knum < kmin) 
        if kmax != None:
            mask = mask | (kmax < knum)

        #frequency cutoff
        if fmin != None:
            mask = mask | (freq < fmin)
        if fmax != None:
            mask = mask | (fmax < freq)

        #dispersion filter
        if hmin != None:
            c = numpy.sqrt(g*hmin)
            omega = 2.*pi*freq/24./3600. / numpy.sqrt(beta*c)           #adusting day^-1 to s^-1
            k     = knum/a * numpy.sqrt(c/beta)                         #adusting ^2pia to ^m               
            mask = mask | (omega**2 - k*omega - 1 < 0)
        if hmax != None:
            c = numpy.sqrt(g*hmax)
            omega = 2.*pi*freq/24./3600. / numpy.sqrt(beta*c)           #adusting day^-1 to s^-1
            k     = knum/a * numpy.sqrt(c/beta)                         #adusting ^2pia to ^m               
            mask = mask | (omega**2 - k*omega - 1 > 0)
        mask = numpy.repeat(mask[:,NA,:], nlat, axis=1)
        fftdata[mask] = 0.0

        filterd = fftpack.ifft2(fftdata, axes=(0,2))
        return filterd.real

    def mrgfilter(self, fmin=None, fmax=None, kmin=-10, kmax=-1, hmin=8, hmax=90):
        """mixed Rossby gravity wave

        Arguments:

           'fmin/fmax' -- unit is cycle per day

           'kmin/kmax' -- zonal wave number. negative is westward, positive is
                          eastward

           'hmin/hmax' -- equivalent depth
        """
        if kmax > 0:
            print("kmax must be negative. if k > 0, this mode is the same as n=0 EIG")
            sys.exit()
            
        fftdata = self.fftdata.copy()
        knum = self.knum
        freq = self.freq
        nf, nlat, nk = fftdata.shape

        # filtering ############################################################
        mask = numpy.zeros((nf,nk), dtype=numpy.bool)
        #wavenumber cut-off
        if kmin != None:
            mask = mask | (knum < kmin) 
        if kmax != None:
            mask = mask | (kmax < knum)

        #frequency cutoff
        if fmin != None:
            mask = mask | (freq < fmin)
        if fmax != None:
            mask = mask | (fmax < freq)

        #dispersion filter
        if hmin != None:
            c = numpy.sqrt(g*hmin)
            omega = 2.*pi*freq/24./3600. / numpy.sqrt(beta*c)           #adusting day^-1 to s^-1
            k     = knum/a * numpy.sqrt(c/beta)                         #adusting ^2pia to ^m               
            mask = mask | (omega**2 - k*omega - 1 < 0)
        if hmax != None:
            c = numpy.sqrt(g*hmax)
            omega = 2.*pi*freq/24./3600. / numpy.sqrt(beta*c)           #adusting day^-1 to s^-1
            k     = knum/a * numpy.sqrt(c/beta)                         #adusting ^2pia to ^m               
            mask = mask | (omega**2 - k*omega - 1 > 0)
        mask = numpy.repeat(mask[:,NA,:], nlat, axis=1)
        fftdata[mask] = 0.0

        filterd = fftpack.ifft2(fftdata, axes=(0,2))
        return filterd.real

    def tdfilter(self, fmin=None, fmax=None, kmin=-20, kmax=-6):
        """KTH05 TD-type filter.

        Arguments:

           'fmin/fmax' -- unit is cycle per day

           'kmin/kmax' -- zonal wave number. negative is westward, positive is
                          eastward
        """
        fftdata = self.fftdata.copy()
        knum = self.knum
        freq = self.freq
        nf, nlat, nk = fftdata.shape
        mask = numpy.zeros((nf,nk), dtype=numpy.bool)

        #wavenumber cut-off
        if kmin != None:
            mask = mask | (knum < kmin) 
        if kmax != None:
            mask = mask | (kmax < knum)

        #frequency cutoff
        if fmin != None:
            mask = mask | (freq < fmin)
        if fmax != None:
            mask = mask | (fmax < freq)

        #dispersion filter
        mask = mask | (84*freq+knum-22 > 0) | (210*freq+2.5*knum-13 < 0)                                                                                         
        mask = numpy.repeat(mask[:,NA,:], nlat, axis=1)

        fftdata[mask] = 0.0

        filterd = fftpack.ifft2(fftdata, axes=(0,2))
        return filterd.real




# from - https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
def DistanceCoord(Lo1,La1,Lo2,La2):

    from math import sin, cos, sqrt, atan2, radians

    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(La1)
    lon1 = radians(Lo1)
    lat2 = radians(La2)
    lon2 = radians(Lo2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance


import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from shapely.ops import unary_union
from shapely.prepared import prep

land_shp_fname = shpreader.natural_earth(resolution='50m',
                                       category='physical', name='land')

land_geom = unary_union(list(shpreader.Reader(land_shp_fname).geometries()))
land = prep(land_geom)

def is_land(x, y):
    return land.contains(sgeom.Point(x, y))


# https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput/33619018#33619018
import numpy as np
from scipy.spatial import ConvexHull

def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T

    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval



# ================================================================================================
# ================================================================================================

def MultiObjectIdentification(
    DATA_all,                      # matrix with data on common grid in the format [time,lat,lon,variable]
                                   # the variables are 'V850' [m/s], 'U850' [m/s], 'T850' K, 'Q850' g/kg, 
                                   # 'SLP' [Pa], 'IVTE' [kg m-1 s-1], 'IVTN' [kg m-1 s-1], 'PR' [mm/time], 'BT' [K]]
                                   # this order must be followed
    Lon,                           # 2D longitude grid centers
    Lat,                           # 2D latitude grid spacing
    Time,                          # datetime vector of data
    dT,                            # integer - temporal frequency of data [hour]
    Mask,                          # mask with dimensions [lat,lon] defining analysis region
    DataName = '',                 # name of the common grid
    OutputFolder='',               # string containing the output directory path. Default is local directory
    # minimum precip. obj.
    SmoothSigmaP = 0,              # Gaussion std for precipitation smoothing
    Pthreshold = 2,                # precipitation threshold [mm/h]
    MinTimePR = 4,                 # minimum lifetime of precip. features in hours
    MinAreaPR = 5000,              # minimum area of precipitation features [km2]
    # minimum Moisture Stream 
    MinTimeMS = 9,                 # minimum lifetime for moisture stream [hours]
    MinAreaMS = 100000,            # mimimum area of moisture stream [km2]
    MinMSthreshold = 0.13,         # treshold for moisture stream [g*m/g*s]
    # cyclone tracking
    MinTimeCY = 12,                # minimum livetime of cyclones [hours]
    MaxPresAnCY = -8,              # preshure thershold for cyclone anomaly [hPa]
    # anty cyclone tracking
    MinTimeACY = 12,               # minimum livetime of anticyclone [hours]
    MinPresAnACY = 6,              # preshure thershold for anti cyclone anomaly [hPa]
    # Frontal zones
    MinAreaFR = 50000,             # mimimum size of frontal zones [km2]
    front_treshold = 1,            # threshold for masking frontal zones
    # Cloud tracking setup
    SmoothSigmaC = 0,              # standard deviation of Gaussian filter for cloud tracking
    Cthreshold = 241,              # brightness temperature threshold for cloud tracking [K]
    MinTimeC = 4,                  # mimimum livetime of ice cloud shields [hours]
    MinAreaC = 40000,              # mimimum area of ice cloud shields [km2]
    # AR tracking
    IVTtrheshold = 500,            # Integrated water vapor transport threshold for AR detection [kg m-1 s-1]
    MinTimeIVT = 9,                # minimum livetime of ARs [hours]
    AR_MinLen = 2000,              # mimimum length of an AR [km]
    AR_Lat = 20,                   # AR centroids have to be poeward of this latitude
    AR_width_lenght_ratio = 2,     # mimimum length to width ratio of AR
    # TC detection
    TC_Pmin = 995,                 # mimimum pressure for TC detection [hPa]
    TC_lat_genesis = 35,           # maximum latitude for TC genesis [absolute degree latitude]
    TC_lat_max = 60,               # maximum latitude for TC existance [absolute degree latitude]
    TC_deltaT_core = 0,            # minimum degrees difference between TC core and surrounding [K]
    TC_T850min = 285,              # minimum temperature of TC core at 850hPa [K]
    TC_minBT = 241,                # minimum average cloud top brightness temperature [K]
    # MCs detection
    MCS_Minsize = 5000,            # minimum size of precipitation area [km2] 
    MCS_minPR = 15,                 # minimum precipitation threshold [mm/h]
    CL_MaxT = 215,                 # minimum brightness temperature in ice shield [K]
    CL_Area = 40000,               # minimum cloud area size [km2]
    MCS_minTime = 4                # minimum lifetime of MCS [hours]
    ):
    
    Variables = ['V850', 'U850', 'T850', 'Q850', 'SLP-1', 'IVTE-1', 'IVTN-1','Z500','V200','U200', 'PR-1', 'BT-1']
    # calculate grid spacing assuming regular lat/lon grid
    _,_,Area,Gridspacing = calc_grid_distance_area(Lon,Lat)
    Area[Area < 0] = 0
    
    
    EarthCircum = 40075000 #[m]
    dLat = np.copy(Lon); dLat[:] = EarthCircum/(360/(Lat[1,0]-Lat[0,0]))
    dLon = np.copy(Lon)
    for la in range(Lat.shape[0]):
        dLon[la,:] = EarthCircum/(360/(Lat[1,0]-Lat[0,0]))*np.cos(np.deg2rad(Lat[la,0]))
#     Gridspacing = np.abs(np.mean(np.append(dLat[:,:,None],dLon[:,:,None], axis=2)))
#     Area = dLat*dLon
#     Area[Area < 0] = 0
    
    rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1
    StartDay = Time[0]
    SetupString = '_dt-'+str(dT)+'h_MOAAP-masks.nc'
    FrontMask = np.copy(Mask)
    FrontMask[np.abs(Lat) < 10] = 0

    # connect over date line?
    if (Lon[0,0] < -176) & (Lon[0,-1] > 176):
        connectLon= 1
    else:
        connectLon= 0


    print('    Derive nescessary varialbes for feature indentification')
    import time
    start = time.perf_counter()
    # 11111111111111111111111111111111111111111111111111
    # calculate vapor transport on pressure level
    VapTrans = ((DATA_all[:,:,:,Variables.index('U850')]*DATA_all[:,:,:,Variables.index('Q850')])**2 + (DATA_all[:,:,:,Variables.index('V850')]*DATA_all[:,:,:,Variables.index('Q850')])**2)**(1/2)

    # 22222222222222222222222222222222222222222222222222
    # Frontal Detection according to https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017GL073662
    UU = DATA_all[:,:,:,Variables.index('U850')]
    VV = DATA_all[:,:,:,Variables.index('V850')]
    dx = dLon
    dy = dLat
    du = np.gradient( UU )
    dv = np.gradient( VV )
    PV = np.abs( dv[-1]/dx[None,:] - du[-2]/dy[None,:] )
    TK = DATA_all[:,:,:,Variables.index('T850')]
    vgrad = np.gradient(TK, axis=(1,2))
    Tgrad = np.sqrt(vgrad[0]**2 + vgrad[1]**2)

    Fstar = PV * Tgrad

    Tgrad_zero = 0.45 #*100/(np.mean([dLon,dLat], axis=0)/1000.)  # 0.45 K/(100 km)
    import metpy.calc as calc
    from metpy.units import units
    CoriolisPar = calc.coriolis_parameter(np.deg2rad(Lat))
    Frontal_Diagnostic = np.array(Fstar/(CoriolisPar * Tgrad_zero))

    # # 3333333333333333333333333333333333333333333333333333
    # # Cyclone identification based on pressure annomaly threshold    
    
    SLP = DATA_all[:,:,:,Variables.index('SLP-1')]/100.
    if np.sum(np.isnan(SLP)) == 0:
        # remove high-frequency variabilities --> smooth over 100 x 100 km (no temporal smoothing)
        SLP_smooth = ndimage.uniform_filter(SLP, size=[1,int(100/(Gridspacing/1000.)),int(100/(Gridspacing/1000.))])
        # smoothign over 3000 x 3000 km and 78 hours
        SLPsmoothAn = ndimage.uniform_filter(SLP, size=[int(78/dT),int(int(3000/(Gridspacing/1000.))),int(int(3000/(Gridspacing/1000.)))])
    else:
        # this code takes care of the smoothing of fields that contain NaN values
        # from - https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
        U=SLP.copy()               # random array...
        V=SLP.copy()
        V[np.isnan(U)]=0
        VV = ndimage.uniform_filter(V, size=[int(78/dT),int(int(3000/(Gridspacing/1000.))),int(int(3000/(Gridspacing/1000.)))])
        W=0*U.copy()+1
        W[np.isnan(U)]=0
        WW=ndimage.uniform_filter(W, size=[int(78/dT),int(int(3000/(Gridspacing/1000.))),int(int(3000/(Gridspacing/1000.)))])
        SLPsmoothAn=VV/WW

        VV = ndimage.uniform_filter(V, size=[1,int(100/(Gridspacing/1000.)),int(100/(Gridspacing/1000.))])
        WW=ndimage.uniform_filter(W, size=[1,int(100/(Gridspacing/1000.)),int(100/(Gridspacing/1000.))])
        SLP_smooth = VV/WW

    SLP_Anomaly = SLP_smooth-SLPsmoothAn
    SLP_Anomaly[:,Mask == 0] = np.nan
    # plt.contour(SLP_Anomaly[tt,:,:], levels=[-9990,-10,1100], colors='b')
    Pressure_anomaly = SLP_Anomaly < MaxPresAnCY # 10 hPa depression | original setting was 12
    HighPressure_annomaly = SLP_Anomaly > MinPresAnACY #12

    # from - https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    sigma=10.0                  # standard deviation for Gaussian kernel
    truncate=10.                # truncate filter at this many sigmas

    U=SLP.copy()               # random array...

    V=SLP.copy()
    V[np.isnan(U)]=0
    VV=gaussian_filter(V,sigma=[sigma,sigma,sigma],truncate=truncate)

    W=0*U.copy()+1
    W[np.isnan(U)]=0
    WW=gaussian_filter(W,sigma=[sigma,sigma,sigma],truncate=truncate)

    Z=VV/WW


    # 4444444444444444444444444444444444444444444444444444444
    # calculate IVT
    IVT = ((DATA_all[:,:,:,Variables.index('IVTE-1')])**2+np.abs(DATA_all[:,:,:,Variables.index('IVTN-1')])**2)**0.5

    # Mask data outside of Focus domain
    DATA_all[:,Mask == 0,:] = np.nan
    Pressure_anomaly[:,Mask == 0] = np.nan
    HighPressure_annomaly[:,Mask == 0] = np.nan
    Frontal_Diagnostic[:,Mask == 0] = np.nan
    VapTrans[:,Mask == 0] = np.nan
    SLP[:,Mask == 0] = np.nan
    
    # 5555555555555555555555555555555555555555555555555555555
    # Detect jet stream features
    uv200 = (DATA_all[:,:,:,Variables.index('U200')]**2 + DATA_all[:,:,:,Variables.index('U200')]**2)**0.5
    
    # 666666666666666666666666666666666666666666666666666666
    # Cyclone and cutoff low detection from 500 hPa Geopotential Heigth (Z500)
    z500 = DATA_all[:,:,:,Variables.index('Z500')]/9.81
    if np.sum(np.isnan(z500)) == 0:
        # remove high-frequency variabilities --> smooth over 100 x 100 km (no temporal smoothing)
        z500_smooth = ndimage.uniform_filter(z500, size=[1,int(100/(Gridspacing/1000.)),int(100/(Gridspacing/1000.))])
        # smoothign over 3000 x 3000 km and 78 hours
        z500smoothAn = ndimage.uniform_filter(z500, size=[int(78/dT),int(int(3000/(Gridspacing/1000.))),int(int(3000/(Gridspacing/1000.)))])
    else:
        # this code takes care of the smoothing of fields that contain NaN values
        # from - https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
        U=z500.copy()               # random array...
        V=z500.copy()
        V[np.isnan(U)]=0
        VV = ndimage.uniform_filter(V, size=[int(78/dT),int(int(3000/(Gridspacing/1000.))),int(int(3000/(Gridspacing/1000.)))])
        W=0*U.copy()+1
        W[np.isnan(U)]=0
        WW=ndimage.uniform_filter(W, size=[int(78/dT),int(int(3000/(Gridspacing/1000.))),int(int(3000/(Gridspacing/1000.)))])
        z500smoothAn=VV/WW

        VV = ndimage.uniform_filter(V, size=[1,int(100/(Gridspacing/1000.)),int(100/(Gridspacing/1000.))])
        WW=ndimage.uniform_filter(W, size=[1,int(100/(Gridspacing/1000.)),int(100/(Gridspacing/1000.))])
        z500_smooth = VV/WW
    
    z500_Anomaly = z500_smooth - z500smoothAn
    z500_Anomaly[:,Mask == 0] = np.nan
    
#     # remove high-frequency variabilities --> smooth over 100 x 100 km (no temporal smoothing)
#     z500_smooth = ndimage.uniform_filter(z500, size=[1,int(100/(Gridspacing/1000.)),int(100/(Gridspacing/1000.))])
#     # smoothign over 3000 x 3000 km and 78 hours
#     z500smoothAn = ndimage.uniform_filter(z500, size=[int(78/dT),int(int(3000/(Gridspacing/1000.))),int(int(3000/(Gridspacing/1000.)))])
#     z500_Anomaly = z500_smooth - z500smoothAn
#     z500_Anomaly[:,Mask == 0] = np.nan

    z_low = z500_Anomaly < -80
    z_high = z500_Anomaly > 70
    
    end = time.perf_counter()
    timer(start, end)


    print('    track  moisture streams in extratropics')
    potARs = (VapTrans > MinMSthreshold)
    rgiObjectsAR, nr_objectsUD = ndimage.label(potARs, structure=rgiObj_Struct)
    print('            '+str(nr_objectsUD)+' object found')

    # sort the objects according to their size
    Objects=ndimage.find_objects(rgiObjectsAR)

    rgiAreaObj = np.array([[np.sum(Area[Objects[ob][1:]][rgiObjectsAR[Objects[ob]][tt,:,:] == ob+1]) for tt in range(rgiObjectsAR[Objects[ob]].shape[0])] for ob in range(nr_objectsUD)])

    # create final object array
    MS_objectsTMP=np.copy(rgiObjectsAR); MS_objectsTMP[:]=0
    ii = 1
    for ob in range(len(rgiAreaObj)):
        AreaTest = np.max(np.convolve(np.array(rgiAreaObj[ob]) >= MinAreaMS*1000**2, np.ones(int(MinTimeMS/dT)), mode='valid'))
        if (AreaTest == int(MinTimeMS/dT)) & (len(rgiAreaObj[ob]) >= int(MinTimeMS/dT)):
            MS_objectsTMP[rgiObjectsAR == (ob+1)] = ii
            ii = ii + 1
    # lable the objects from 1 to N
    # MS_objects=np.copy(rgiObjectsAR); MS_objects[:]=0
    # Unique = np.unique(MS_objectsTMP)[1:]
    # ii = 1
    # for ob in range(len(Unique)):
    #     MS_objects[MS_objectsTMP == Unique[ob]] = ii
    #     ii = ii + 1

    MS_objects, _ = clean_up_objects(MS_objectsTMP,
                                  dT,
                               min_tsteps=0)

    print('        break up long living MS objects that have many elements')
    MS_objects, object_split = BreakupObjects(MS_objects,
                                int(MinTimeMS/dT),
                                dT)

    if connectLon == 1:
        print('        connect MS objects over date line')
        MS_objects = ConnectLon_on_timestep(MS_objects)


    grMSs = calc_object_characteristics(MS_objects, # feature object file
                                 VapTrans,         # original file used for feature detection
                                 OutputFolder+'MS850_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,        # output file name and locaiton
                                 Time,            # timesteps of the data
                                 Lat,             # 2D latidudes
                                 Lon,             # 2D Longitudes
                                 Gridspacing,
                                 Area,
                                 min_tsteps=int(MinTimeMS/dT))      # minimum livetime in hours


    #### ------------------------
    print('    track  IVT')
    import time
    start = time.perf_counter()

    potIVTs = (IVT > IVTtrheshold)
    rgiObjectsIVT, nr_objectsUD = ndimage.label(potIVTs, structure=rgiObj_Struct)
    print('        '+str(nr_objectsUD)+' object found')

    # # sort the objects according to their size
    # Objects=ndimage.find_objects(rgiObjectsIVT)
    # # rgiVolObj=np.array([np.sum(rgiObjectsIVT[Objects[ob]] == ob+1) for ob in range(nr_objectsUD)])
    # TT_CY = np.array([Objects[ob][0].stop - Objects[ob][0].start for ob in range(nr_objectsUD)])

    # # create final object array
    # IVT_objectsTMP=np.copy(rgiObjectsIVT); IVT_objectsTMP[:]=0
    # ii = 1
    # for ob in range(len(TT_CY)):
    #     if TT_CY[ob] >= int(MinTimeIVT/dT):
    #         IVT_objectsTMP[rgiObjectsIVT == (ob+1)] = ii
    #         ii = ii + 1
    # # lable the objects from 1 to N
    # IVT_objects=np.copy(rgiObjectsIVT); IVT_objects[:]=0
    # Unique = np.unique(IVT_objectsTMP)[1:]
    # ii = 1
    # for ob in range(len(Unique)):
    #     IVT_objects[IVT_objectsTMP == Unique[ob]] = ii
    #     ii = ii + 1

    IVT_objects, _ = clean_up_objects(rgiObjectsIVT,
                                   dT,
                            min_tsteps=int(MinTimeIVT/dT))

    print('        break up long living IVT objects that have many elements')
    IVT_objects, object_split = BreakupObjects(IVT_objects,
                                 int(MinTimeIVT/dT),
                                dT)

    if connectLon == 1:
        print('        connect IVT objects over date line')
        IVT_objects = ConnectLon_on_timestep(IVT_objects)

    grIVTs = calc_object_characteristics(IVT_objects, # feature object file
                                 IVT,         # original file used for feature detection
                                 OutputFolder+'IVT_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                                 Time,            # timesteps of the data
                                 Lat,             # 2D latidudes
                                 Lon,             # 2D Longitudes
                                 Gridspacing,
                                 Area,
                                 min_tsteps=int(MinTimeIVT/dT))      # minimum livetime in hours
    end = time.perf_counter()
    timer(start, end)

    print('    check if MSs quallify as ARs')
    start = time.perf_counter()
    if IVT_objects.max() != 0:
        AR_obj = np.copy(IVT_objects); AR_obj[:] = 0.
        Objects=ndimage.find_objects(IVT_objects.astype(int))
    else:
        AR_obj = np.copy(MS_objects); AR_obj[:] = 0.
        Objects=ndimage.find_objects(MS_objects.astype(int))
        IVT_objects = MS_objects
    aa=1
    for ii in range(len(Objects)):
        if Objects[ii] == None:
            continue
        ObjACT = IVT_objects[Objects[ii]] == ii+1
        LonObj = Lon[Objects[ii][1],Objects[ii][2]]
        LatObj = Lat[Objects[ii][1],Objects[ii][2]]
        # check if object crosses the date line
        if LonObj.max()-LonObj.min() > 359:
            ObjACT = np.roll(ObjACT, int(ObjACT.shape[2]/2), axis=2)

        OBJ_max_len = np.zeros((ObjACT.shape[0]))
        for tt in range(ObjACT.shape[0]):
            PointsObj = np.append(LonObj[ObjACT[tt,:,:]==1][:,None], LatObj[ObjACT[tt,:,:]==1][:,None], axis=1)
            try:
                Hull = scipy.spatial.ConvexHull(np.array(PointsObj))
            except:
                continue
            XX = []; YY=[]
            for simplex in Hull.simplices:
    #                 plt.plot(PointsObj[simplex, 0], PointsObj[simplex, 1], 'k-')
                XX = XX + [PointsObj[simplex, 0][0]] 
                YY = YY + [PointsObj[simplex, 1][0]]

            points = [[XX[ii],YY[ii]] for ii in range(len(YY))]
            BOX = minimum_bounding_rectangle(np.array(PointsObj))

            DIST = np.zeros((3))
            for rr in range(3):
                DIST[rr] = DistanceCoord(BOX[rr][0],BOX[rr][1],BOX[rr+1][0],BOX[rr+1][1])
            OBJ_max_len[tt] = np.max(DIST)
            if OBJ_max_len[tt] <= AR_MinLen:
                ObjACT[tt,:,:] = 0
            else:
                rgiCenter = np.round(ndimage.measurements.center_of_mass(ObjACT[tt,:,:])).astype(int)
                LatCent = LatObj[rgiCenter[0],rgiCenter[1]]
                if np.abs(LatCent) < AR_Lat:
                    ObjACT[tt,:,:] = 0
            # check width to lenght ratio
            if DIST.max()/DIST.min() < AR_width_lenght_ratio:
                ObjACT[tt,:,:] = 0
        if LonObj.max()-LonObj.min() > 359:
            ObjACT = np.roll(ObjACT, -int(ObjACT.shape[2]/2), axis=2)
        ObjACT = ObjACT.astype(int)
        ObjACT[ObjACT!=0] = aa
        ObjACT = ObjACT + AR_obj[Objects[ii]]
        AR_obj[Objects[ii]] = ObjACT
        aa=aa+1

    grACs = calc_object_characteristics(AR_obj, # feature object file
                         IVT,         # original file used for feature detection
                         OutputFolder+'ARs_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                         Time,            # timesteps of the data
                         Lat,             # 2D latidudes
                         Lon,             # 2D Longitudes
                         Gridspacing,
                         Area)
    end = time.perf_counter()
    timer(start, end)

    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    print('    track cyclones')
    start = time.perf_counter()
    #     Pressure_anomaly[np.isnan(Pressure_anomaly)] = 0
    Pressure_anomaly[:,Mask == 0] = 0
    rgiObjectsUD, nr_objectsUD = ndimage.label(Pressure_anomaly,structure=rgiObj_Struct)
    print('            '+str(nr_objectsUD)+' object found')

    # # sort the objects according to their size
    # Objects=ndimage.find_objects(rgiObjectsUD)
    # rgiVolObj=np.array([np.sum(rgiObjectsUD[Objects[ob]] == ob+1) for ob in range(nr_objectsUD)])
    # TT_CY = np.array([Objects[ob][0].stop - Objects[ob][0].start for ob in range(nr_objectsUD)])

    # # create final object array
    # CY_objectsTMP=np.copy(rgiObjectsUD); CY_objectsTMP[:]=0
    # ii = 1
    # for ob in range(len(rgiVolObj)):
    #     if TT_CY[ob] >= int(MinTimeCY/dT):
    #         CY_objectsTMP[rgiObjectsUD == (ob+1)] = ii
    #         ii = ii + 1

    # # lable the objects from 1 to N
    # CY_objects=np.copy(rgiObjectsUD); CY_objects[:]=0
    # Unique = np.unique(CY_objectsTMP)[1:]
    # ii = 1
    # for ob in range(len(Unique)):
    #     CY_objects[CY_objectsTMP == Unique[ob]] = ii
    #     ii = ii + 1

    CY_objects, _ = clean_up_objects(rgiObjectsUD,
                                  dT,
                            min_tsteps=int(MinTimeCY/dT))


    print('        break up long living CY objects that heve many elements')
    CY_objects, object_split = BreakupObjects(CY_objects,
                                int(MinTimeCY/dT),
                                dT)
    if connectLon == 1:
        print('        connect cyclones objects over date line')
        CY_objects = ConnectLon_on_timestep(CY_objects)


    grCyclonesPT = calc_object_characteristics(CY_objects, # feature object file
                                     SLP,         # original file used for feature detection
                                     OutputFolder+'CY_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                                     Time,            # timesteps of the data
                                     Lat,             # 2D latidudes
                                     Lon,             # 2D Longitudes
                                     Gridspacing,
                                     Area,
                                     min_tsteps=int(MinTimeCY/dT)) 
    end = time.perf_counter()
    timer(start, end)



    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    print('    track anti-cyclones')
    start = time.perf_counter()
    HighPressure_annomaly[:,Mask == 0] = 0
    rgiObjectsUD, nr_objectsUD = ndimage.label(HighPressure_annomaly,structure=rgiObj_Struct)
    print('        '+str(nr_objectsUD)+' object found')

    # # sort the objects according to their size
    # Objects=ndimage.find_objects(rgiObjectsUD)
    # rgiVolObj=np.array([np.sum(rgiObjectsUD[Objects[ob]] == ob+1) for ob in range(nr_objectsUD)])
    # TT_ACY = np.array([Objects[ob][0].stop - Objects[ob][0].start for ob in range(nr_objectsUD)])

    # # create final object array
    # ACY_objectsTMP=np.copy(rgiObjectsUD); ACY_objectsTMP[:]=0
    # ii = 1
    # for ob in range(len(rgiVolObj)):
    #     if TT_ACY[ob] >= MinTimeACY:
    # #     if rgiVolObj[ob] >= MinVol:
    #         ACY_objectsTMP[rgiObjectsUD == (ob+1)] = ii
    #         ii = ii + 1

    # # lable the objects from 1 to N
    # ACY_objects=np.copy(rgiObjectsUD); ACY_objects[:]=0
    # Unique = np.unique(ACY_objectsTMP)[1:]
    # ii = 1
    # for ob in range(len(Unique)):
    #     ACY_objects[ACY_objectsTMP == Unique[ob]] = ii
    #     ii = ii + 1

    ACY_objects, _ = clean_up_objects(rgiObjectsUD,
                                   dT,
                            min_tsteps=int(MinTimeACY/dT))


    print('        break up long living ACY objects that have many elements')
    ACY_objects, object_split = BreakupObjects(ACY_objects,
                                int(MinTimeCY/dT),
                                dT)
    if connectLon == 1:
        # connect objects over date line
        ACY_objects = ConnectLon_on_timestep(ACY_objects)

    grACyclonesPT = calc_object_characteristics(ACY_objects, # feature object file
                                     SLP,         # original file used for feature detection
                                     OutputFolder+'ACY_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                                     Time,            # timesteps of the data
                                     Lat,             # 2D latidudes
                                     Lon,             # 2D Longitudes
                                     Gridspacing,
                                     Area,
                                     min_tsteps=int(MinTimeCY/dT)) 
    end = time.perf_counter()
    timer(start, end)


    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    print('    track 500 hPa anticyclones')
    start = time.perf_counter()
    #     Pressure_anomaly[np.isnan(Pressure_anomaly)] = 0
    z_high[:,Mask == 0] = 0
    rgiObjectsUD, nr_objectsUD = ndimage.label(z_high, structure=rgiObj_Struct)
    print('            '+str(nr_objectsUD)+' object found')

    acy_z500_objects, _ = clean_up_objects(rgiObjectsUD,
                                min_tsteps=int(MinTimeCY/dT),
                                 dT = dT)


    print('        break up long living CY objects that heve many elements')
    acy_z500_objects, object_split = BreakupObjects(acy_z500_objects,
                                int(MinTimeCY/dT),
                                dT)
    if connectLon == 1:
        print('        connect cyclones objects over date line')
        acy_z500_objects = ConnectLon_on_timestep(acy_z500_objects)


    acy_z500_objects_characteristics = calc_object_characteristics(acy_z500_objects, # feature object file
                                     z500,         # original file used for feature detection
                                     OutputFolder+'CY-z500_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                                     Time,            # timesteps of the data
                                     Lat,             # 2D latidudes
                                     Lon,             # 2D Longitudes
                                     Gridspacing,
                                     Area,
                                     min_tsteps=int(MinTimeCY/dT)) 
    end = time.perf_counter()
    timer(start, end)
    
    
    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    print('    track 500 hPa cyclones')
    start = time.perf_counter()
    #     Pressure_anomaly[np.isnan(Pressure_anomaly)] = 0
    z_low[:,Mask == 0] = 0
    rgiObjectsUD, nr_objectsUD = ndimage.label(z_low,structure=rgiObj_Struct)
    print('            '+str(nr_objectsUD)+' object found')

    cy_z500_objects, _ = clean_up_objects(rgiObjectsUD,
                                min_tsteps=int(MinTimeCY/dT),
                                 dT = dT)


    print('        break up long living CY objects that heve many elements')
    cy_z500_objects, object_split = BreakupObjects(cy_z500_objects,
                                int(MinTimeCY/dT),
                                dT)
    if connectLon == 1:
        print('        connect cyclones objects over date line')
        cy_z500_objects = ConnectLon_on_timestep(cy_z500_objects)


    cy_z500_objects_characteristics = calc_object_characteristics(cy_z500_objects, # feature object file
                                     z500,         # original file used for feature detection
                                     OutputFolder+'CY-z500_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                                     Time,            # timesteps of the data
                                     Lat,             # 2D latidudes
                                     Lon,             # 2D Longitudes
                                     Gridspacing,
                                     Area,
                                     min_tsteps=int(MinTimeCY/dT)) 
    end = time.perf_counter()
    timer(start, end)
    
    
    # Check if cyclones are cutoff lows
    # ------------------------
    # ------------------------
    # ------------------------
    # ------------------------
    # ------------------------
    # ------------------------
    print('    Check if cyclones qualify as Cut Off Low (COL)')
    ### COL detection is similar to https://journals.ametsoc.org/view/journals/clim/33/6/jcli-d-19-0497.1.xml
    start = time.perf_counter()
    col_Tracks = {}
    col_Time = {}
    aa=1
    # area arround cyclone
    col_buffer = 500000 # m

    # check if cyclone is COL
    Objects=ndimage.find_objects(cy_z500_objects.astype(int))
    col_obj = np.copy(cy_z500_objects); col_obj[:]=0
    u200 = DATA_all[:,:,:,Variables.index('U200')]
    for ii in range(len(Objects)):
        if Objects[ii] == None:
            continue
        ObjACT = cy_z500_objects[Objects[ii]] == ii+1
        if ObjACT.shape[0] < MinTimeC:
            continue

        dxObj = abs(np.mean(dx[Objects[ii][1],Objects[ii][2]]))
        dyObj = abs(np.mean(dy[Objects[ii][1],Objects[ii][2]]))
        col_buffer_obj_lo = int(col_buffer/dxObj)
        col_buffer_obj_la = int(col_buffer/dyObj)

        # add buffer to object slice
        tt_start = Objects[ii][0].start
        tt_stop = Objects[ii][0].stop
        lo_start = Objects[ii][2].start - col_buffer_obj_lo 
        lo_stop = Objects[ii][2].stop + col_buffer_obj_lo
        la_start = Objects[ii][1].start - col_buffer_obj_la 
        la_stop = Objects[ii][1].stop + col_buffer_obj_la
        if lo_start < 0:
            lo_start = 0
        if lo_stop >= Lon.shape[1]:
            lo_stop = Lon.shape[1]-1
        if la_start < 0:
            la_start = 0
        if la_stop >= Lon.shape[0]:
            la_stop = Lon.shape[0]-1

        LonObj = Lon[la_start:la_stop, lo_start:lo_stop]
        LatObj = Lat[la_start:la_stop, lo_start:lo_stop]

        z500_ACT = np.copy(z500[tt_start:tt_stop, la_start:la_stop, lo_start:lo_stop])
        ObjACT = cy_z500_objects[tt_start:tt_stop, la_start:la_stop, lo_start:lo_stop] == ii+1
        u200_ob = u200[tt_start:tt_stop, la_start:la_stop, lo_start:lo_stop]
        front_ob = Frontal_Diagnostic[tt_start:tt_stop, la_start:la_stop, lo_start:lo_stop]
        if LonObj[0,-1] - LonObj[0,0] > 358:
            sift_lo = 'yes'
            # object crosses the date line
            shift = int(LonObj.shape[1]/2)
            LonObj = np.roll(LonObj, shift, axis=1)
            LatObj = np.roll(LatObj, shift, axis=1)
            z500_ACT = np.roll(z500_ACT, shift, axis=2)
            ObjACT = np.roll(ObjACT, shift, axis=2)
            u200_ob = np.roll(u200_ob, shift, axis=2)
            front_ob = np.roll(front_ob, shift, axis=2)
        else:
            sift_lo = 'no'

        # find location of z500 minimum
        z500_ACT_obj = np.copy(z500_ACT)
        z500_ACT_obj[ObjACT == 0] = 999999999999.

        for tt in range(z500_ACT_obj.shape[0]):
            min_loc = np.where(z500_ACT_obj[tt,:,:] == np.nanmin(z500_ACT_obj[tt]))
            min_la = min_loc[0][0]
            min_lo = min_loc[1][0]
            la_0 = min_la - col_buffer_obj_la
            if la_0 < 0:
                la_0 = 0
            lo_0 = min_lo - col_buffer_obj_lo
            if lo_0 < 0:
                lo_0 = 0

            lat_reg = LatObj[la_0:min_la + col_buffer_obj_la+1,
                             lo_0:min_lo + col_buffer_obj_lo+1]
            lon_reg = LonObj[la_0:min_la + col_buffer_obj_la+1,
                             lo_0:min_lo + col_buffer_obj_lo+1]

            col_region = z500_ACT[tt,
                                  la_0:min_la + col_buffer_obj_la+1,
                                  lo_0:min_lo + col_buffer_obj_lo+1]
            obj_col_region = z500_ACT_obj[tt,
                                  la_0:min_la + col_buffer_obj_la+1,
                                  lo_0:min_lo + col_buffer_obj_lo+1]
            min_z500_obj = z500_ACT[tt,min_la,min_lo]
            u200_ob_region = u200_ob[tt,
                                  la_0:min_la + col_buffer_obj_la+1,
                                  lo_0:min_lo + col_buffer_obj_lo+1]
            front_ob_region = front_ob[tt,
                                  la_0:min_la + col_buffer_obj_la+1,
                                  lo_0:min_lo + col_buffer_obj_lo+1]


            # check if 350 km radius arround center has higher Z
            min_loc_tt = np.where(obj_col_region[:,:] == 
                                  np.nanmin(z500_ACT_obj[tt]))
            min_la_tt = min_loc_tt[0][0]
            min_lo_tt = min_loc_tt[1][0]

            rdist = radialdistance(lat_reg[min_la_tt,min_lo_tt],
                                   lon_reg[min_la_tt,min_lo_tt],
                                   lat_reg,
                                   lon_reg)

            # COL should only occure between 20 and 70 degrees
            # https://journals.ametsoc.org/view/journals/clim/33/6/jcli-d-19-0497.1.xml
            if (abs(lat_reg[min_la_tt,min_lo_tt]) < 20) | (abs(lat_reg[min_la_tt,min_lo_tt]) > 70):
                ObjACT[tt,:,:] = 0
                continue

            # remove cyclones that are close to the poles
            if np.max(np.abs(lat_reg)) > 88:
                ObjACT[tt,:,:] = 0
                continue

            if np.nanmin(z500_ACT_obj[tt]) > 100000:
                # there is no object to process
                ObjACT[tt,:,:] = 0
                continue

            # CRITERIA 1) at least 75 % of grid cells in ring have have 10 m higher Z than center
            ring = (rdist >= (350 - (dxObj/1000.)*2))  & (rdist <= (350 + (dxObj/1000.)*2))
            if np.sum((min_z500_obj - col_region[ring]) < -10) < np.sum(ring)*0.75:
                ObjACT[tt,:,:] = 0
                continue

            # CRITERIA 2) check if 200 hPa wind speed is eastward in the poleward direction of the cyclone
            if lat_reg[min_la_tt,min_lo_tt] > 0:
                east_flow = u200_ob_region[0 : min_la_tt,
                                    min_lo_tt]
            else:
                east_flow = u200_ob_region[min_la_tt : -1,
                                    min_lo_tt]

            try:
                if np.min(east_flow) > 0:
                    ObjACT[tt,:,:] = 0
                    continue
            except:
                ObjACT[tt,:,:] = 0
                continue

            # Criteria 3) frontal zone in eastern flank of COL
            front_test = np.sum(np.abs(front_ob_region[:, min_lo_tt:]) > 1)
            if front_test < 1:
                ObjACT[tt,:,:] = 0
                continue

        if sift_lo == 'yes':
            ObjACT = np.roll(ObjACT, -shift, axis=2)

        ObjACT = ObjACT.astype('int')
        ObjACT[ObjACT > 0] = ii+1
        ObjACT = ObjACT + col_obj[tt_start:tt_stop, la_start:la_stop, lo_start:lo_stop]
        col_obj[tt_start:tt_stop, la_start:la_stop, lo_start:lo_stop] = ObjACT

    col_stats = calc_object_characteristics(col_obj, # feature object file
                         DATA_all[:,:,:,Variables.index('Z500')],         # original file used for feature detection
                         OutputFolder+'COL_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                         Time,            # timesteps of the data
                         Lat,             # 2D latidudes
                         Lon,             # 2D Longitudes
                         Gridspacing,
                         Area,
                         min_tsteps=1)      # minimum livetime in hours
    end = time.perf_counter()
    timer(start, end)
    
    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------
    print('    track jetstream')

    MinTimeJS = 24 # hours
    js_min_anomaly = 22

    
    if np.sum(np.isnan(uv200)) == 0:
        uv200_smooth = ndimage.uniform_filter(uv200, size=[1,int(500/(Gridspacing/1000.)),int(500/(Gridspacing/1000.))])
        # smoothign over 3000 x 3000 km and 78 hours
        uv200smoothAn = ndimage.uniform_filter(uv200, size=[int(78/dT),int(int(5000/(Gridspacing/1000.))),int(int(5000/(Gridspacing/1000.)))])
    else:
        # this code takes care of the smoothing of fields that contain NaN values
        # from - https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
        U=uv200.copy()               # random array...
        V=uv200.copy()
        V[np.isnan(U)]=0
        VV = ndimage.uniform_filter(V, size=[int(78/dT),int(int(5000/(Gridspacing/1000.))),int(int(5000/(Gridspacing/1000.)))])
        W=0*U.copy()+1
        W[np.isnan(U)]=0
        WW=ndimage.uniform_filter(W, size=[int(78/dT),int(int(5000/(Gridspacing/1000.))),int(int(5000/(Gridspacing/1000.)))])
        uv200smoothAn=VV/WW

        VV = ndimage.uniform_filter(V, size=[1,int(500/(Gridspacing/1000.)),int(500/(Gridspacing/1000.))])
        WW=ndimage.uniform_filter(W, size=[1,int(500/(Gridspacing/1000.)),int(500/(Gridspacing/1000.))])
        uv200_smooth = VV/WW
    
    uv200_Anomaly = uv200_smooth - uv200smoothAn
    jet = uv200_Anomaly[:,:,:] >= js_min_anomaly


    start = time.perf_counter()
    #     Pressure_anomaly[np.isnan(Pressure_anomaly)] = 0
    jet[:,Mask == 0] = 0
    rgiObjectsUD, nr_objectsUD = ndimage.label(jet, structure=rgiObj_Struct)
    print('            '+str(nr_objectsUD)+' object found')

    jet_objects, _ = clean_up_objects(rgiObjectsUD,
                                min_tsteps=int(MinTimeJS/dT),
                                 dT = dT)


    print('        break up long living CY objects that heve many elements')
    jet_objects, object_split = BreakupObjects(jet_objects,
                                int(MinTimeCY/dT),
                                dT)
    if connectLon == 1:
        print('        connect cyclones objects over date line')
        jet_objects = ConnectLon_on_timestep(jet_objects)


    jet_objects_characteristics = calc_object_characteristics(jet_objects, # feature object file
                                     uv200,         # original file used for feature detection
                                     OutputFolder+'jet_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                                     Time,            # timesteps of the data
                                     Lat,             # 2D latidudes
                                     Lon,             # 2D Longitudes
                                     Gridspacing,
                                     Area,
                                     min_tsteps=int(MinTimeCY/dT)) 
    end = time.perf_counter()
    timer(start, end)
    
    
    ### Identify Frontal regions
    # ------------------------
    print('    identify frontal zones')
    start = time.perf_counter()
    rgiObj_Struct_Fronts=np.zeros((3,3,3)); rgiObj_Struct_Fronts[1,:,:]=1

    Frontal_Diagnostic = np.abs(Frontal_Diagnostic)
    Frontal_Diagnostic[:,FrontMask == 0] = 0
    Fmask = (Frontal_Diagnostic > front_treshold)

    rgiObjectsUD, nr_objectsUD = ndimage.label(Fmask,structure=rgiObj_Struct_Fronts)
    print('        '+str(nr_objectsUD)+' object found')

    # # calculate object size
    Objects=ndimage.find_objects(rgiObjectsUD)
    rgiAreaObj = np.array([np.sum(Area[Objects[ob][1:]][rgiObjectsUD[Objects[ob]][0,:,:] == ob+1]) for ob in range(nr_objectsUD)])

    # rgiAreaObj=np.array([np.sum(rgiObjectsUD[Objects[ob]] == ob+1) for ob in range(nr_objectsUD)])
    # create final object array
    FR_objects=np.copy(rgiObjectsUD)
    TooSmall = np.where(rgiAreaObj < MinAreaFR*1000**2)
    FR_objects[np.isin(FR_objects, TooSmall[0]+1)] = 0

    #     FR_objects=np.copy(rgiObjectsUD); FR_objects[:]=0
    #     ii = 1
    #     for ob in range(len(rgiAreaObj)):
    #         if rgiAreaObj[ob] >= MinAreaFR*1000**2:
    #             FR_objects[rgiObjectsUD == (ob+1)] = ii
    #             ii = ii + 1
    end = time.perf_counter()
    timer(start, end)


    # ------------------------
    print('    track  precipitation')
    start = time.perf_counter()
    PRsmooth=gaussian_filter(DATA_all[:,:,:,Variables.index('PR-1')], sigma=(0,SmoothSigmaP,SmoothSigmaP))
    PRmask = (PRsmooth >= Pthreshold*dT)
    rgiObjectsPR, nr_objectsUD = ndimage.label(PRmask, structure=rgiObj_Struct)
    print('        '+str(nr_objectsUD)+' precipitation object found')

    if connectLon == 1:
        # connect objects over date line
        rgiObjectsPR = ConnectLon(rgiObjectsPR)

    # remove None objects
    Objects=ndimage.find_objects(rgiObjectsPR)
    rgiVolObj=np.array([np.sum(rgiObjectsPR[Objects[ob]] == ob+1) for ob in range(nr_objectsUD)])
    ZERO_V =  np.where(rgiVolObj == 0)
    if len(ZERO_V[0]) > 0:
        Dummy = (slice(0, 1, None), slice(0, 1, None), slice(0, 1, None))
        Objects = np.array(Objects)
        for jj in ZERO_V[0]:
            Objects[jj] = Dummy

    # Remove objects that are too small or short lived
    rgiAreaObj = np.array([[np.sum(Area[Objects[ob][1],Objects[ob][2]][rgiObjectsPR[Objects[ob]][tt,:,:] == ob+1]) for tt in range(rgiObjectsPR[Objects[ob]].shape[0])] for ob in range(nr_objectsUD)])
    # create final object array
    PR_objects=np.copy(rgiObjectsPR); PR_objects[:]=0
    ii = 1
    for ob in range(len(rgiAreaObj)):
        AreaTest = np.max(np.convolve(np.array(rgiAreaObj[ob]) >= MinAreaPR*1000**2, np.ones(int(MinTimePR/dT)), mode='valid'))
        if (AreaTest == int(MinTimePR/dT)) & (len(rgiAreaObj[ob]) >= int(MinTimePR/dT)):
            PR_objects[rgiObjectsPR == (ob+1)] = ii
            ii = ii + 1

    print('        break up long living precipitation objects that have many elements')
    PR_objects, object_split = BreakupObjects(PR_objects,
                                int(MinTimePR/dT),
                                dT)

    if connectLon == 1:
        print('        connect precipitation objects over date line')
        PR_objects = ConnectLon_on_timestep(PR_objects)

    grPRs = calc_object_characteristics(PR_objects, # feature object file
                                 DATA_all[:,:,:,Variables.index('PR-1')],         # original file used for feature detection
                                 OutputFolder+'PR_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                                 Time,            # timesteps of the data
                                 Lat,             # 2D latidudes
                                 Lon,             # 2D Longitudes
                                 Gridspacing,
                                 Area,
                                 min_tsteps=int(MinTimePR/dT))      # minimum livetime in hours

    end = time.perf_counter()
    timer(start, end)


    # ------------------------
    print('    track  clouds')
    start = time.perf_counter()
    Csmooth=gaussian_filter(DATA_all[:,:,:,Variables.index('BT-1')], sigma=(0,SmoothSigmaC,SmoothSigmaC))
    Cmask = (Csmooth <= Cthreshold)
    rgiObjectsC, nr_objectsUD = ndimage.label(Cmask, structure=rgiObj_Struct)
    print('        '+str(nr_objectsUD)+' cloud object found')

    if connectLon == 1:
        # connect objects over date line
        rgiObjectsC = ConnectLon(rgiObjectsC)

    # minimum cloud volume
    # sort the objects according to their size
    Objects=ndimage.find_objects(rgiObjectsC)

    rgiAreaObj = np.array([[np.sum(Area[Objects[ob][1],Objects[ob][2]][rgiObjectsC[Objects[ob]][tt,:,:] == ob+1]) for tt in range(rgiObjectsC[Objects[ob]].shape[0])] for ob in range(nr_objectsUD)])

    # rgiVolObjC=np.array([np.sum(rgiObjectsC[Objects[ob]] == ob+1) for ob in range(nr_objectsUD)])

    # create final object array
    C_objects=np.copy(rgiObjectsC); C_objects[:]=0
    ii = 1
    for ob in range(len(rgiAreaObj)):
        AreaTest = np.max(np.convolve(np.array(rgiAreaObj[ob]) >= MinAreaC*1000**2, np.ones(int(MinTimeC/dT)), mode='valid'))
        if (AreaTest == int(MinTimeC/dT)) & (len(rgiAreaObj[ob]) >=int(MinTimeC/dT)):
        # if rgiVolObjC[ob] >= MinAreaC:
            C_objects[rgiObjectsC == (ob+1)] = ii
            ii = ii + 1

    print('        break up long living cloud shield objects that have many elements')
    C_objects, object_split = BreakupObjects(C_objects,
                                int(MinTimeC/dT),
                                dT)

    if connectLon == 1:
        print('        connect cloud objects over date line')
        C_objects = ConnectLon_on_timestep(C_objects)

    grCs = calc_object_characteristics(C_objects, # feature object file
                                 DATA_all[:,:,:,Variables.index('BT-1')],         # original file used for feature detection
                                 OutputFolder+'Clouds_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                                 Time,            # timesteps of the data
                                 Lat,             # 2D latidudes
                                 Lon,             # 2D Longitudes
                                 Gridspacing,
                                 Area,
                                 min_tsteps=int(MinTimeC/dT))      # minimum livetime in hours
    end = time.perf_counter()
    timer(start, end)

    
    

    # ------------------------
    print('    check if pr objects quallify as MCS')
    start = time.perf_counter()
    # check if precipitation object is from an MCS
    Objects=ndimage.find_objects(PR_objects.astype(int))
    MCS_obj = np.copy(PR_objects); MCS_obj[:]=0
    window_length = int(MCS_minTime/dT)
    for ii in range(len(Objects)):
        if Objects[ii] == None:
            continue
        ObjACT = PR_objects[Objects[ii]] == ii+1
        if ObjACT.shape[0] < 2:
            continue
        if ObjACT.shape[0] < window_length:
            continue
        Cloud_ACT = np.copy(C_objects[Objects[ii]])
        LonObj = Lon[Objects[ii][1],Objects[ii][2]]
        LatObj = Lat[Objects[ii][1],Objects[ii][2]]   
        Area_ACT = Area[Objects[ii][1],Objects[ii][2]]
        PR_ACT = DATA_all[:,:,:,Variables.index('PR-1')][Objects[ii]]

        PR_Size = np.array([np.sum(Area_ACT[ObjACT[tt,:,:] >0]) for tt in range(ObjACT.shape[0])])
        PR_MAX = np.array([np.max(PR_ACT[tt,ObjACT[tt,:,:] >0]) if len(PR_ACT[tt,ObjACT[tt,:,:]>0]) > 0 else 0 for tt in range(ObjACT.shape[0])])
        # Get cloud shield
        rgiCL_obj = np.delete(np.unique(Cloud_ACT[ObjACT > 0]),0)
        if len(rgiCL_obj) == 0:
            # no deep cloud shield is over the precipitation
            continue
        CL_OB_TMP = C_objects[Objects[ii][0]]
        CLOUD_obj_act = np.in1d(CL_OB_TMP.flatten(), rgiCL_obj).reshape(CL_OB_TMP.shape)
        Cloud_Size = np.array([np.sum(Area[CLOUD_obj_act[tt,:,:] >0]) for tt in range(CLOUD_obj_act.shape[0])])
        # min temperatur must be taken over precip area
    #     CL_ob_pr = C_objects[Objects[ii]]
        CL_BT_pr = np.copy(DATA_all[:,:,:,Variables.index('BT-1')][Objects[ii]])
        CL_BT_pr[ObjACT == 0] = np.nan
        Cloud_MinT = np.nanmin(CL_BT_pr, axis=(1,2))
    #     Cloud_MinT = np.array([np.min(CL_BT_pr[tt,CL_ob_pr[tt,:,:] >0]) if len(CL_ob_pr[tt,CL_ob_pr[tt,:,:] >0]) > 0 else 0 for tt in range(CL_ob_pr.shape[0])])
        Cloud_MinT[Cloud_MinT < 150 ] = np.nan
        # is precipitation associated with AR?
        AR_ob = np.copy(AR_obj[Objects[ii]])
        AR_ob[:,LatObj < 25] = 0 # only consider ARs in mid- and hight latitudes
        AR_test = np.sum(AR_ob > 0, axis=(1,2))            

        MCS_max_residence = np.min([int(24/dT),ObjACT.shape[0]]) # MCS criterion must be meet within this time window
                               # or MCS is discontinued
        # minimum lifetime peak precipitation
        is_pr_peak_intense = np.convolve(
                                        PR_MAX >= MCS_minPR*dT, 
                                        np.ones(MCS_max_residence), 'same') >= 1
        # minimum precipitation area threshold
        is_pr_size = np.convolve(
                        (np.convolve((PR_Size / 1000**2 >= MCS_Minsize), np.ones(window_length), 'same') / window_length) == 1, 
                                        np.ones(MCS_max_residence), 'same') >= 1
        # Tb size and time threshold
        is_Tb_area = np.convolve(
                        (np.convolve((Cloud_Size / 1000**2 >= CL_Area), np.ones(window_length), 'same') / window_length) == 1, 
                                        np.ones(MCS_max_residence), 'same') >= 1
        # Tb overshoot
        is_Tb_overshoot = np.convolve(
                            Cloud_MinT  <= CL_MaxT, 
                            np.ones(MCS_max_residence), 'same') >= 1
        try:
            MCS_test = (
                        (is_pr_peak_intense == 1)
                        & (is_pr_size == 1)
                        & (is_Tb_area == 1)
                        & (is_Tb_overshoot == 1)
                )
            ObjACT[MCS_test == 0,:,:] = 0
        except:
            ObjACT[MCS_test == 0,:,:] = 0

        

        # assign unique object numbers
        ObjACT = np.array(ObjACT).astype(int)
        ObjACT[ObjACT == 1] = ii+1

    #         # remove all precip that is associated with ARs
    #         ObjACT[AR_test > 0] = 0

    #     # PR area defines MCS area and precipitation
    #     window_length = int(MCS_minTime/dT)
    #     cumulative_sum = np.cumsum(np.insert(MCS_TEST, 0, 0))
    #     moving_averages = (cumulative_sum[window_length:] - cumulative_sum[:-window_length]) / window_length
    #     if ob == 16:
    #         stop()
        if np.max(MCS_test) == 1:
            TMP = np.copy(MCS_obj[Objects[ii]])
            TMP = TMP + ObjACT
            MCS_obj[Objects[ii]] = TMP
        else:
            continue
    
    MCS_obj, _ = clean_up_objects(MCS_obj,
                                   dT,
                            min_tsteps=int(MCS_minTime/dT))        
    
    grMCSs = calc_object_characteristics(MCS_obj, # feature object file
                             DATA_all[:,:,:,Variables.index('PR-1')],         # original file used for feature detection
                             OutputFolder+'MCSs_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                             Time,            # timesteps of the data
                             Lat,             # 2D latidudes
                             Lon,             # 2D Longitudes
                             Gridspacing,
                             Area)
    
    end = time.perf_counter()
    timer(start, end)
    
    
    ###########################################################
    ###########################################################
    # TRACK MCSs AS TB SHIELDS
    print(f"======> check if Tb objects quallify as MCS (or selected storm type)")
    start_time = time.time()
    bt_data = DATA_all[:,:,:,Variables.index('BT-1')]
    # check if precipitation object is from an MCS
    object_indices = ndimage.find_objects(rgiObjectsC)
    MCS_objects_Tb = np.zeros(rgiObjectsC.shape, dtype=int)
    
    

    for iobj,_ in tqdm(enumerate(object_indices)):
        if object_indices[iobj] is None:
            continue

        time_slice = object_indices[iobj][0]
        lat_slice  = object_indices[iobj][1]
        lon_slice  = object_indices[iobj][2]

        tb_object_slice= rgiObjectsC[object_indices[iobj]]
        tb_object_act = np.where(tb_object_slice==iobj+1,True,False)
        if len(tb_object_act) < MCS_minTime:
            continue

        tb_slice =  bt_data[object_indices[iobj]]
        tb_act = np.copy(tb_slice)
        tb_act[~tb_object_act] = np.nan

        bt_object_slice = rgiObjectsC[object_indices[iobj]]
        bt_object_act = np.copy(bt_object_slice)
        bt_object_act[~tb_object_act] = 0

        area_act = np.tile(Area[lat_slice, lon_slice], (tb_act.shape[0], 1, 1))
        area_act[~tb_object_act] = 0

        ### Calculate cloud properties
        tb_size = np.array(np.sum(area_act,axis=(1,2)))
        tb_min = np.array(np.nanmin(tb_act,axis=(1,2)))

        ### Calculate precipitation properties
        pr_act = np.copy(DATA_all[:,:,:,Variables.index('PR-1')][object_indices[iobj]])
        pr_act[tb_object_act == 0] = np.nan

        pr_peak_act = np.array(np.nanmax(pr_act,axis=(1,2)))

        pr_region_act = pr_act >= Pthreshold*dT
        area_act = np.tile(Area[lat_slice, lon_slice], (tb_act.shape[0], 1, 1))
        area_act[~pr_region_act] = 0
        pr_under_cloud = np.array(np.sum(area_act,axis=(1,2)))/1000**2 


        # Test if object classifies as MCS
        tb_size_test = np.max(np.convolve((tb_size / 1000**2 >= CL_Area), np.ones(MCS_minTime), 'valid') / MCS_minTime) == 1
        tb_overshoot_test = np.max((tb_min  <= CL_MaxT )) == 1
        pr_peak_test = np.max(np.convolve((pr_peak_act >= MCS_minPR ), np.ones(MCS_minTime), 'valid') / MCS_minTime) ==1
        pr_area_test = np.max((pr_under_cloud >= MCS_Minsize)) == 1
        MCS_test = (
                    tb_size_test
                    & tb_overshoot_test
                    & pr_peak_test
                    & pr_area_test
        )

        # assign unique object numbers
        tb_object_act = np.array(tb_object_act).astype(int)
        tb_object_act[tb_object_act == 1] = iobj + 1

#         window_length = int(MCS_minTime / dT)
#         moving_averages = np.convolve(MCS_test, np.ones(window_length), 'valid') / window_length

    #     if iobj+1 == 19:
    #         stop()
        if MCS_test == 1:
            TMP = np.copy(MCS_objects_Tb[object_indices[iobj]])
            TMP = TMP + tb_object_act
            MCS_objects_Tb[object_indices[iobj]] = TMP

        else:
            continue
    end_time = time.time()
    print(f"======> 'Calculate cloud characteristics: {(end_time-start_time):.2f} seconds \n")
    start_time = time.time()

    MCS_objects_Tb, _ = clean_up_objects(MCS_objects_Tb,
                                       dT,
                                       min_tsteps=int(MCS_minTime/dT))  


    #if len(objects_overlap)>1: import pdb; pdb.set_trace()
    # objects_id_MCS, num_objects = ndimage.label(MCS_objects_Tb, structure=obj_structure_3D)
    grMCSs_Tb = calc_object_characteristics(
        MCS_objects_Tb,  # feature object file
        DATA_all[:,:,:,Variables.index('PR-1')],  # original file used for feature detection
        OutputFolder+'MCSs_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString+'_Tb.pkl',
        Time,            # timesteps of the data
        Lat,             # 2D latidudes
        Lon,             # 2D Longitudes
        Gridspacing,
        Area)

    end_time = time.time()
    print(f"======> 'MCS Tb tracking: {(end_time-start_time):.2f} seconds \n")
    start_time = time.time()


    # ------------------------
    # ------------------------
    # ------------------------
    # ------------------------
    # ------------------------
    # ------------------------
    print('    Check if cyclones qualify as TCs')
    start = time.perf_counter()
    TC_Tracks = {}
    TC_Time = {}
#     aa=1
    # check if cyclone is tropical
    Objects=ndimage.find_objects(CY_objects.astype(int))
    TC_obj = np.copy(CY_objects); TC_obj[:]=0
    for ii in range(len(Objects)):
        if Objects[ii] == None:
            continue
        ObjACT = CY_objects[Objects[ii]] == ii+1
        if ObjACT.shape[0] < 2*8:
            continue
        T_ACT = np.copy(TK[Objects[ii]])
        SLP_ACT = np.copy(SLP[Objects[ii]])
        LonObj = Lon[Objects[ii][1],Objects[ii][2]]
        LatObj = Lat[Objects[ii][1],Objects[ii][2]]
        # check if object crosses the date line
        if LonObj.max()-LonObj.min() > 359:
            ObjACT = np.roll(ObjACT, int(ObjACT.shape[2]/2), axis=2)
            SLP_ACT = np.roll(SLP_ACT, int(ObjACT.shape[2]/2), axis=2)
        # Calculate low pressure center track
        SLP_ACT[ObjACT == 0] = 999999999.
        Track_ACT = np.array([np.argwhere(SLP_ACT[tt,:,:] == np.nanmin(SLP_ACT[tt,:,:]))[0] for tt in range(ObjACT.shape[0])])
        LatLonTrackAct = np.array([(LatObj[Track_ACT[tt][0],Track_ACT[tt][1]],LonObj[Track_ACT[tt][0],Track_ACT[tt][1]]) for tt in range(ObjACT.shape[0])])
        if np.min(np.abs(LatLonTrackAct[:,0])) > TC_lat_genesis:
            ObjACT[:] = 0
            continue
        else:

            # has the cyclone a warm core?
            DeltaTCore = np.zeros((ObjACT.shape[0])); DeltaTCore[:] = np.nan
            T850_core = np.copy(DeltaTCore)
            for tt in range(ObjACT.shape[0]):
                T_cent = np.mean(T_ACT[tt,Track_ACT[tt,0]-1:Track_ACT[tt,0]+2,Track_ACT[tt,1]-1:Track_ACT[tt,1]+2])
                T850_core[tt] = T_cent
                T_Cyclone = np.mean(T_ACT[tt,ObjACT[tt,:,:] != 0])
    #                     T_Cyclone = np.mean(T_ACT[tt,MassC[0]-5:MassC[0]+6,MassC[1]-5:MassC[1]+6])
                DeltaTCore[tt] = T_cent-T_Cyclone
            # smooth the data
            DeltaTCore = gaussian_filter(DeltaTCore,1)
            WarmCore = DeltaTCore > TC_deltaT_core

            if np.sum(WarmCore) < 8:
                continue
            ObjACT[WarmCore == 0,:,:] = 0
            # is the core temperature warm enough
            ObjACT[T850_core < TC_T850min,:,:] = 0


            # TC must have pressure of less 980 hPa
            MinPress = np.min(SLP_ACT, axis=(1,2))
            if np.sum(MinPress < TC_Pmin) < 8:
                continue

            # is the cloud shield cold enough?
            PR_objACT = np.copy(PR_objects[Objects[ii]])
            BT_act = np.copy(DATA_all[:,:,:,Variables.index('BT-1')][Objects[ii]])
            BT_objMean = np.zeros((BT_act.shape[0])); BT_objMean[:] = np.nan
            for tt in range(len(BT_objMean)):
                try:
                    BT_objMean[tt] = np.nanmean(BT_act[tt,PR_objACT[tt,:,:] != 0])
                except:
                    continue

            # is cloud shild overlapping with TC?
            BT_objACT = np.copy(C_objects[Objects[ii]])
            bt_overlap = np.array([np.sum((BT_objACT[kk,ObjACT[kk,:,:] == True] > 0) == True)/np.sum(ObjACT[10,:,:] == True) for kk in range(ObjACT.shape[0])]) > 0.4

        # remove pieces of the track that are not TCs
        TCcheck = (T850_core > TC_T850min) & (WarmCore == 1) & (MinPress < TC_Pmin) #& (bt_overlap == 1) #(BT_objMean < TC_minBT)
        LatLonTrackAct[TCcheck == False,:] = np.nan

        Max_LAT = (np.abs(LatLonTrackAct[:,0]) >  TC_lat_max)
        LatLonTrackAct[Max_LAT,:] = np.nan

        if np.sum(~np.isnan(LatLonTrackAct[:,0])) == 0:
            continue

        # check if cyclone genesis is over water; each re-emergence of TC is a new genesis
        resultLAT = [list(map(float,g)) for k,g in groupby(LatLonTrackAct[:,0], np.isnan) if not k]
        resultLON = [list(map(float,g)) for k,g in groupby(LatLonTrackAct[:,1], np.isnan) if not k]
        LS_genesis = np.zeros((len(resultLAT))); LS_genesis[:] = np.nan
        for jj in range(len(resultLAT)):
            LS_genesis[jj] = is_land(resultLON[jj][0],resultLAT[jj][0])
        if np.max(LS_genesis) == 1:
            for jj in range(len(LS_genesis)):
                if LS_genesis[jj] == 1:
                    SetNAN = np.isin(LatLonTrackAct[:,0],resultLAT[jj])
                    LatLonTrackAct[SetNAN,:] = np.nan

        # make sure that only TC time slizes are considered
        ObjACT[np.isnan(LatLonTrackAct[:,0]),:,:] = 0

        if LonObj.max()-LonObj.min() > 359:
            ObjACT = np.roll(ObjACT, -int(ObjACT.shape[2]/2), axis=2)
        ObjACT = ObjACT.astype(int)
        ObjACT[ObjACT!=0] = ii+1

        ObjACT = ObjACT + TC_obj[Objects[ii]]
        TC_obj[Objects[ii]] = ObjACT
        TC_Tracks[str(ii+1)] = LatLonTrackAct
#         aa=aa+1
        
#     TC_obj, _ = clean_up_objects(TC_obj,
#                                    dT)   
    
    grTCs = calc_object_characteristics(TC_obj, # feature object file
                             DATA_all[:,:,:,Variables.index('SLP-1')],         # original file used for feature detection
                             OutputFolder+'TC_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                             Time,            # timesteps of the data
                             Lat,             # 2D latidudes
                             Lon,             # 2D Longitudes
                             Gridspacing,
                             Area,
                             min_tsteps=int(MinTimeC/dT))      # minimum livetime in hours

    end = time.perf_counter()
    timer(start, end)

    print(' ')
    print('Save the object masks into a joint netCDF')
    start = time.perf_counter()
    # ============================
    # Write NetCDF
    iTime = np.array((Time - Time[0]).total_seconds()).astype('int')

    dataset = Dataset(NCfile,'w',format='NETCDF4_CLASSIC')
    yc = dataset.createDimension('yc', Lat.shape[0])
    xc = dataset.createDimension('xc', Lat.shape[1])
    time = dataset.createDimension('time', None)

    times = dataset.createVariable('time', np.float64, ('time',))
    lat = dataset.createVariable('lat', np.float32, ('yc','xc',))
    lon = dataset.createVariable('lon', np.float32, ('yc','xc',))
    PR_real = dataset.createVariable('PR', np.float32,('time','yc','xc'),zlib=True)
    PR_obj = dataset.createVariable('PR_Objects', np.float32,('time','yc','xc'),zlib=True)
    MCSs = dataset.createVariable('MCS_Objects', np.float32,('time','yc','xc'),zlib=True)
    MCSs_Tb = dataset.createVariable('MCS_Tb_Objects', np.float32,('time','yc','xc'),zlib=True)
    Cloud_real = dataset.createVariable('BT', np.float32,('time','yc','xc'),zlib=True)
    Cloud_obj = dataset.createVariable('BT_Objects', np.float32,('time','yc','xc'),zlib=True)
    FR_real = dataset.createVariable('FR', np.float32,('time','yc','xc'),zlib=True)
    FR_obj = dataset.createVariable('FR_Objects', np.float32,('time','yc','xc'),zlib=True)
    CY_real = dataset.createVariable('CY', np.float32,('time','yc','xc'),zlib=True)
    CY_obj = dataset.createVariable('CY_Objects', np.float32,('time','yc','xc'),zlib=True)
    TCs = dataset.createVariable('TC_Objects', np.float32,('time','yc','xc'),zlib=True)
    ACY_obj = dataset.createVariable('ACY_Objects', np.float32,('time','yc','xc'),zlib=True)
    MS_real = dataset.createVariable('MS', np.float32,('time','yc','xc'),zlib=True)
    MS_obj = dataset.createVariable('MS_Objects', np.float32,('time','yc','xc'),zlib=True)
    IVT_real = dataset.createVariable('IVT', np.float32,('time','yc','xc'),zlib=True)
    IVT_obj = dataset.createVariable('IVT_Objects', np.float32,('time','yc','xc'),zlib=True)
    ARs = dataset.createVariable('AR_Objects', np.float32,('time','yc','xc'),zlib=True)
    SLP_real = dataset.createVariable('SLP', np.float32,('time','yc','xc'),zlib=True)
    T_real = dataset.createVariable('T850', np.float32,('time','yc','xc'),zlib=True)
    CY_z500_real = dataset.createVariable('CY_z500', np.float32,('time','yc','xc'),zlib=True)
    CY_z500_obj = dataset.createVariable('CY_z500_Objects', np.float32,('time','yc','xc'),zlib=True)
    ACY_z500_obj = dataset.createVariable('ACY_z500_Objects', np.float32,('time','yc','xc'),zlib=True)
    Z500_real = dataset.createVariable('Z500', np.float32,('time','yc','xc'),zlib=True)
    COL = dataset.createVariable('COL_Objects', np.float32,('time','yc','xc'),zlib=True)
    JET = dataset.createVariable('JET_Objects', np.float32,('time','yc','xc'),zlib=True)

    times.calendar = "standard"
    times.units = "seconds since "+str(Time[0].year)+"-"+str(Time[0].month).zfill(2)+"-"+str(Time[0].day).zfill(2)+" "+str(Time[0].hour).zfill(2)+":"+str(Time[0].minute).zfill(2)+":00"
    times.standard_name = "time"
    times.long_name = "time"

    lat.long_name = "latitude" ;
    lat.units = "degrees_north" ;
    lat.standard_name = "latitude" ;

    lon.long_name = "longitude" ;
    lon.units = "degrees_east" ;
    lon.standard_name = "longitude" ;

    PR_real.coordinates = "lon lat"
    PR_obj.coordinates = "lon lat"
    MCSs.coordinates = "lon lat"
    MCSs_Tb.coordinates = "lon lat"
    FR_real.coordinates = "lon lat"
    FR_obj.coordinates = "lon lat"
    CY_real.coordinates = "lon lat"
    CY_obj.coordinates = "lon lat"
    ACY_obj.coordinates = "lon lat"
    SLP_real.coordinates = "lon lat"
    T_real.coordinates = "lon lat"
    Cloud_real.coordinates = "lon lat"
    Cloud_obj.coordinates = "lon lat"
    MS_real.coordinates = "lon lat"
    MS_obj.coordinates = "lon lat"
    IVT_real.coordinates = "lon lat"
    IVT_obj.coordinates = "lon lat"
    ARs.coordinates = "lon lat"
    TCs.coordinates = "lon lat"
    CY_z500_real.coordinates = "lon lat"
    CY_z500_obj.coordinates = "lon lat"
    ACY_z500_obj.coordinates = "lon lat"
    COL.coordinates = "lon lat"
    Z500_real.coordinates = "lon lat"
    JET.coordinates = "lon lat"

    lat[:] = Lat
    lon[:] = Lon
    PR_real[:] = DATA_all[:,:,:,Variables.index('PR-1')]
    PR_obj[:] = PR_objects
    MCSs[:] = MCS_obj
    MCSs_Tb[:] = MCS_objects_Tb
    FR_real[:] = Frontal_Diagnostic
    FR_obj[:] = FR_objects
    CY_real[:] = SLP_Anomaly
    CY_obj[:] = CY_objects
    TCs[:] = TC_obj
    ACY_obj[:] = ACY_objects
    SLP_real[:] = SLP
    T_real[:] = TK
    MS_real[:] = VapTrans
    MS_obj[:] = MS_objects
    IVT_real[:] = IVT
    IVT_obj[:] = IVT_objects
    ARs[:] = AR_obj
    Cloud_real[:] = DATA_all[:,:,:,Variables.index('BT-1')]
    Cloud_obj[:] = C_objects
    CY_z500_real[:] = z500
    CY_z500_obj[:] = cy_z500_objects
    ACY_z500_obj[:] = acy_z500_objects
    COL[:] = col_obj
    Z500_real[:] = DATA_all[:,:,:,Variables.index('Z500')]
    JET[:] = jet_objects
    times[:] = iTime

    dataset.close()
    print('Saved: '+NCfile)
    import time
    end = time.perf_counter()
    timer(start, end)

    ### SAVE THE TC TRACKS TO PICKL FILE
    # ============================
    a_file = open(OutputFolder+str(Time[0].year)+str(Time[0].month).zfill(2)+'_TCs_tracks.pkl', "wb")
    pickle.dump(TC_Tracks, a_file)
    a_file.close()
    # CYCLONES[YYYY+MM] = TC_Tracks
    

    
    
    
    
############################################################
###########################################################
#### ======================================================
def MCStracking(
    pr_data,
    bt_data,
    times,
    Lon,
    Lat,
    nc_file,
    DataOutDir,
    DataName):
    """ Function to track MCS from precipitation and brightness temperature
    """

    import mcs_config as cfg
    from skimage.measure import regionprops
    start_time = time.time()
    #Reading tracking parameters

    DT = cfg.DT

    #Precipitation tracking setup
    smooth_sigma_pr = cfg.smooth_sigma_pr   # [0] Gaussion std for precipitation smoothing
    thres_pr        = cfg.thres_pr     # [2] precipitation threshold [mm/h]
    min_time_pr     = cfg.min_time_pr     # [3] minum lifetime of PR feature in hours
    min_area_pr     = cfg.min_area_pr      # [5000] minimum area of precipitation feature in km2
    # Brightness temperature (Tb) tracking setup
    smooth_sigma_bt = cfg.smooth_sigma_bt   #  [0] Gaussion std for Tb smoothing
    thres_bt        = cfg.thres_bt     # [241] minimum Tb of cloud shield
    min_time_bt     = cfg.min_time_bt       # [9] minium lifetime of cloud shield in hours
    min_area_bt     = cfg.min_area_bt       # [40000] minimum area of cloud shield in km2
    # MCs detection
    MCS_min_pr_MajorAxLen  = cfg.MCS_min_pr_MajorAxLen    # [100] km | minimum length of major axis of precipitation object
    MCS_thres_pr       = cfg.MCS_thres_pr      # [10] minimum max precipitation in mm/h
    MCS_thres_peak_pr   = cfg.MCS_thres_peak_pr  # [10] Minimum lifetime peak of MCS precipitation
    MCS_thres_bt     = cfg.MCS_thres_bt        # [225] minimum brightness temperature
    MCS_min_area_bt         = cfg.MCS_min_area_bt        # [40000] min cloud area size in km2
    MCS_min_time     = cfg.MCS_min_time    # [4] minimum time step


    #     DT = 1                    # temporal resolution of data for tracking in hours

    #     # MINIMUM REQUIREMENTS FOR FEATURE DETECTION
    #     # precipitation tracking options
    #     smooth_sigma_pr = 0          # Gaussion std for precipitation smoothing
    #     thres_pr = 2            # precipitation threshold [mm/h]
    #     min_time_pr = 3             # minum lifetime of PR feature in hours
    #     min_area_pr = 5000          # minimum area of precipitation feature in km2

    #     # Brightness temperature (Tb) tracking setup
    #     smooth_sigma_bt = 0          # Gaussion std for Tb smoothing
    #     thres_bt = 241          # minimum Tb of cloud shield
    #     min_time_bt = 9              # minium lifetime of cloud shield in hours
    #     min_area_bt = 40000          # minimum area of cloud shield in km2

    #     # MCs detection
    #     MCS_min_area = min_area_pr   # minimum area of MCS precipitation object in km2
    #     MCS_thres_pr = 10            # minimum max precipitation in mm/h
    #     MCS_thres_peak_pr = 10        # Minimum lifetime peak of MCS precipitation
    #     MCS_thres_bt = 225             # minimum brightness temperature
    #     MCS_min_area_bt = MinAreaC        # min cloud area size in km2
    #     MCS_min_time = 4           # minimum lifetime of MCS

    #Calculating grid distances and areas

    _,_,grid_cell_area,grid_spacing = calc_grid_distance_area(Lon,Lat)
    grid_cell_area[grid_cell_area < 0] = 0

    obj_structure_3D = np.ones((3,3,3))

    start_day = times[0]


    # connect over date line?
    crosses_dateline = False
    if (Lon[0, 0] < -176) & (Lon[0, -1] > 176):
        crosses_dateline = True

    end_time = time.time()
    print(f"======> 'Initialize MCS tracking function: {(end_time-start_time):.2f} seconds \n")
    start_time = time.time()
    # --------------------------------------------------------
    # TRACKING PRECIP OBJECTS
    # --------------------------------------------------------
    print("        track  precipitation")

    pr_smooth= filters.gaussian_filter(
        pr_data, sigma=(0, smooth_sigma_pr, smooth_sigma_pr)
    )
    pr_mask = pr_smooth >= thres_pr * DT
    objects_id_pr, num_objects = ndimage.label(pr_mask, structure=obj_structure_3D)
    print("            " + str(num_objects) + " precipitation object found")

    # connect objects over date line
    if crosses_dateline:
        objects_id_pr = ConnectLon(objects_id_pr)

    # get indices of object to reduce memory requirements during manipulation
    object_indices = ndimage.find_objects(objects_id_pr)


    #Calcualte area of objects
    area_objects = calculate_area_objects(objects_id_pr,object_indices,grid_cell_area)

    # Keep only large and long enough objects
    # Remove objects that are too small or short lived
    pr_objects = remove_small_short_objects(objects_id_pr,area_objects,min_area_pr,min_time_pr,DT, objects = object_indices)

    grPRs = calc_object_characteristics(
        pr_objects,  # feature object file
        pr_data,  # original file used for feature detection
        DataOutDir+DataName+"_PR_"+str(start_day.year)+str(start_day.month).zfill(2)+'.pkl',
        times,  # timesteps of the data
        Lat,  # 2D latidudes
        Lon,  # 2D Longitudes
        grid_spacing,
        grid_cell_area,
        min_tsteps=int(min_time_pr/ DT), # minimum lifetime in data timesteps
    )

    end_time = time.time()
    print(f"======> 'Tracking precip: {(end_time-start_time):.2f} seconds \n")
    start_time = time.time()
    # --------------------------------------------------------
    # TRACKING CLOUD (BT) OBJECTS
    # --------------------------------------------------------
    print("            track  clouds")
    bt_smooth = filters.gaussian_filter(
        bt_data, sigma=(0, smooth_sigma_bt, smooth_sigma_bt)
    )
    bt_mask = bt_smooth <= thres_bt
    objects_id_bt, num_objects = ndimage.label(bt_mask, structure=obj_structure_3D)
    print("            " + str(num_objects) + " cloud object found")

    # connect objects over date line
    if crosses_dateline:
        print("            connect cloud objects over date line")
        objects_id_bt = ConnectLon(objects_id_bt)

    # get indices of object to reduce memory requirements during manipulation
    object_indices = ndimage.find_objects(objects_id_bt)

    #Calcualte area of objects
    area_objects = calculate_area_objects(objects_id_bt,object_indices,grid_cell_area)

    # Keep only large and long enough objects
    # Remove objects that are too small or short lived
    objects_id_bt = remove_small_short_objects(objects_id_bt,area_objects,min_area_bt,min_time_bt,DT, objects = object_indices)

    end_time = time.time()
    print(f"======> 'Tracking clouds: {(end_time-start_time):.2f} seconds \n")
    start_time = time.time()

    print("            break up long living cloud shield objects that have many elements")
    objects_id_bt, object_split = BreakupObjects(objects_id_bt, int(min_time_bt / DT), DT)

    end_time = time.time()
    print(f"======> 'Breaking up cloud objects: {(end_time-start_time):.2f} seconds \n")
    start_time = time.time()

    grCs = calc_object_characteristics(
        objects_id_bt,  # feature object file
        bt_data,  # original file used for feature detection
        DataOutDir+DataName+"_BT_"+str(start_day.year)+str(start_day.month).zfill(2)+'.pkl',
        times,  # timesteps of the data
        Lat,  # 2D latidudes
        Lon,  # 2D Longitudes
        grid_spacing,
        grid_cell_area,
        min_tsteps=int(min_time_bt / DT), # minimum lifetime in data timesteps
    )
    end_time = time.time()
    print(f"======> 'Calculate cloud characteristics: {(end_time-start_time):.2f} seconds \n")
    start_time = time.time()
    # --------------------------------------------------------
    # CHECK IF PR OBJECTS QUALIFY AS MCS
    # (or selected strom type according to msc_config.py)
    # --------------------------------------------------------
    print("            check if pr objects quallify as MCS (or selected storm type)")
    # check if precipitation object is from an MCS
    object_indices = ndimage.find_objects(pr_objects)
    MCS_objects = np.zeros(pr_objects.shape,dtype=int)

    for iobj,_ in enumerate(object_indices):
        if object_indices[iobj] is None:
            continue

        time_slice = object_indices[iobj][0]
        lat_slice  = object_indices[iobj][1]
        lon_slice  = object_indices[iobj][2]


        pr_object_slice= pr_objects[object_indices[iobj]]
        pr_object_act = np.where(pr_object_slice==iobj+1,True,False)

        if len(pr_object_act) < 2:
            continue

        pr_slice =  pr_data[object_indices[iobj]]
        pr_act = np.copy(pr_slice)
        pr_act[~pr_object_act] = 0

        bt_slice  = bt_data[object_indices[iobj]]
        bt_act = np.copy(bt_slice)
        bt_act[~pr_object_act] = 0

        bt_object_slice = objects_id_bt[object_indices[iobj]]
        bt_object_act = np.copy(bt_object_slice)
        bt_object_act[~pr_object_act] = 0

        area_act = np.tile(grid_cell_area[lat_slice, lon_slice], (pr_act.shape[0], 1, 1))
        area_act[~pr_object_act] = 0

    #     pr_size = np.array(np.sum(area_act,axis=(1,2)))
        pr_max = np.array(np.max(pr_act,axis=(1,2)))

        # calculate major axis length of PR object
        pr_object_majoraxislen = np.array([
                regionprops(pr_object_act[tt,:,:].astype(int))[0].major_axis_length*np.mean(area_act[tt,(pr_object_act[tt,:,:] == 1)]/1000**2)**0.5 
                for tt in range(pr_object_act.shape[0])
            ])

        #Check overlaps between clouds (bt) and precip objects
        objects_overlap = np.delete(np.unique(bt_object_act[pr_object_act]),0)

        if len(objects_overlap) == 0:
            # no deep cloud shield is over the precipitation
            continue

        ## Keep bt objects (entire) that partially overlap with pr object

        bt_object_overlap = np.in1d(objects_id_bt[time_slice].flatten(), objects_overlap).reshape(objects_id_bt[time_slice].shape)

        # Get size of all cloud (bt) objects together
        # We get size of all cloud objects that overlap partially with pr object
        # DO WE REALLY NEED THIS?

        bt_size = np.array(
            [
            np.sum(grid_cell_area[bt_object_overlap[tt, :, :] > 0])
            for tt in range(bt_object_overlap.shape[0])
            ]
        )

        #Check if BT is below threshold over precip areas
        bt_min_temp = np.nanmin(np.where(bt_object_slice>0,bt_slice,999),axis=(1,2))

        # minimum lifetime peak precipitation
        is_pr_peak_intense = np.max(pr_max) >= MCS_thres_peak_pr * DT
        MCS_test = (
                    (bt_size / 1000**2 >= MCS_min_area_bt)
                    & (np.sum(bt_min_temp  <= MCS_thres_bt ) > 0)
                    & (pr_object_majoraxislen >= MCS_min_pr_MajorAxLen )
                    & (pr_max >= MCS_thres_pr * DT)
                    & (is_pr_peak_intense)
        )

        # assign unique object numbers

        pr_object_act = np.array(pr_object_act).astype(int)
        pr_object_act[pr_object_act == 1] = iobj + 1

        window_length = int(MCS_min_time / DT)
        moving_averages = np.convolve(MCS_test, np.ones(window_length), 'valid') / window_length

    #     if iobj+1 == 19:
    #         stop()

        if (len(moving_averages) > 0) & (np.max(moving_averages) == 1):
            TMP = np.copy(MCS_objects[object_indices[iobj]])
            TMP = TMP + pr_object_act
            MCS_objects[object_indices[iobj]] = TMP

        else:
            continue

    #if len(objects_overlap)>1: import pdb; pdb.set_trace()
    # objects_id_MCS, num_objects = ndimage.label(MCS_objects, structure=obj_structure_3D)
    grMCSs = calc_object_characteristics(
        MCS_objects,  # feature object file
        pr_data,  # original file used for feature detection
        DataOutDir+DataName+"_MCS_"+str(start_day.year)+str(start_day.month).zfill(2)+'.pkl',
        times,  # timesteps of the data
        Lat,  # 2D latidudes
        Lon,  # 2D Longitudes
        grid_spacing,
        grid_cell_area,
        min_tsteps=int(MCS_min_time / DT), # minimum lifetime in data timesteps
    )

    end_time = time.time()
    print(f"======> 'MCS tracking: {(end_time-start_time):.2f} seconds \n")
    start_time = time.time()
    

    ###########################################################
    ###########################################################
    ## WRite netCDF with xarray
    if nc_file is not None:
        print ('Save objects into a netCDF')

        fino=xr.Dataset({'MCS_objects':(['time','y','x'],MCS_objects),
                         'PR':(['time','y','x'],pr_data),
                         'PR_objects':(['time','y','x'],objects_id_pr),
                         'BT':(['time','y','x'],bt_data),
                         'BT_objects':(['time','y','x'],objects_id_bt),
                         'lat':(['y','x'],Lat),
                         'lon':(['y','x'],Lon)},
                         coords={'time':times.values})

        fino.to_netcdf(nc_file,mode='w',encoding={'PR':{'zlib': True,'complevel': 5},
                                                 'PR_objects':{'zlib': True,'complevel': 5},
                                                 'BT':{'zlib': True,'complevel': 5},
                                                 'BT_objects':{'zlib': True,'complevel': 5},
                                                 'MCS_objects':{'zlib': True,'complevel': 5}})


    # fino = xr.Dataset({
    # 'MCS_objects': xr.DataArray(
    #             data   = objects_id_MCS,   # enter data here
    #             dims   = ['time','y','x'],
    #             attrs  = {
    #                 '_FillValue': const.missingval,
    #                 'long_name': 'Mesoscale Convective System objects',
    #                 'units'     : '',
    #                 }
    #             ),
    # 'PR_objects': xr.DataArray(
    #             data   = objects_id_pr,   # enter data here
    #             dims   = ['time','y','x'],
    #             attrs  = {
    #                 '_FillValue': const.missingval,
    #                 'long_name': 'Precipitation objects',
    #                 'units'     : '',
    #                 }
    #             ),
    # 'BT_objects': xr.DataArray(
    #             data   = objects_id_bt,   # enter data here
    #             dims   = ['time','y','x'],
    #             attrs  = {
    #                 '_FillValue': const.missingval,
    #                 'long_name': 'Cloud (brightness temperature) objects',
    #                 'units'     : '',
    #                 }
    #             ),
    # 'PR': xr.DataArray(
    #             data   = pr_data,   # enter data here
    #             dims   = ['time','y','x'],
    #             attrs  = {
    #                 '_FillValue': const.missingval,
    #                 'long_name': 'Precipitation',
    #                 'standard_name': 'precipitation',
    #                 'units'     : 'mm h-1',
    #                 }
    #             ),
    # 'BT': xr.DataArray(
    #             data   = bt_data,   # enter data here
    #             dims   = ['time','y','x'],
    #             attrs  = {
    #                 '_FillValue': const.missingval,
    #                 'long_name': 'Brightness temperature',
    #                 'standard_name': 'brightness_temperature',
    #                 'units'     : 'K',
    #                 }
    #             ),
    # 'lat': xr.DataArray(
    #             data   = Lat,   # enter data here
    #             dims   = ['y','x'],
    #             attrs  = {
    #                 '_FillValue': const.missingval,
    #                 'long_name': "latitude",
    #                 'standard_name': "latitude",
    #                 'units'     : "degrees_north",
    #                 }
    #             ),
    # 'lon': xr.DataArray(
    #             data   = Lon,   # enter data here
    #             dims   = ['y','x'],
    #             attrs  = {
    #                 '_FillValue': const.missingval,
    #                 'long_name': "longitude",
    #                 'standard_name': "longitude",
    #                 'units'     : "degrees_east",
    #                 }
    #             ),
    #         },
    #     attrs = {'date':datetime.date.today().strftime('%Y-%m-%d'),
    #              "comments": "File created with MCS_tracking"},
    #     coords={'time':times.values}
    # )


    # fino.to_netcdf(nc_file,mode='w',format = "NETCDF4",
    #                encoding={'PR':{'zlib': True,'complevel': 5},
    #                          'PR_objects':{'zlib': True,'complevel': 5},
    #                          'BT':{'zlib': True,'complevel': 5},
    #                          'BT_objects':{'zlib': True,'complevel': 5}})


        end_time = time.time()
        print(f"======> 'Writing files: {(end_time-start_time):.2f} seconds \n")
        start_time = time.time()
    else:
        print(f"No writing files required, output file name is empty")
    ###########################################################
    ###########################################################
    # ============================
    # Write NetCDF
    return grMCSs, MCS_objects



def clean_up_objects(DATA,
                     dT,
                     min_tsteps = 0,
                     obj_splitmerge = None):
    """ Function to remove objects that are too short lived
        and to numerrate the object from 1...N
    """
    
    object_indices = ndimage.find_objects(DATA)
    MaxOb = np.max(DATA)
    MinLif = int(24 / dT)  # min lifetime of object to be split
    AVmax = 1.5

    id_translate = np.zeros((len(object_indices),2))
    objectsTMP = np.copy(DATA)
    objectsTMP[:] = 0
    ii = 1
    for obj in range(len(object_indices)):
        if object_indices[obj] != None:
            if object_indices[obj][0].stop - object_indices[obj][0].start >= min_tsteps / dT:
                Obj_tmp = np.copy(objectsTMP[object_indices[obj]])
                Obj_tmp[DATA[object_indices[obj]] == obj+1] = ii
                objectsTMP[object_indices[obj]] = Obj_tmp
                id_translate[obj,0] = obj+1
                id_translate[obj,1] = ii
                ii = ii + 1
            else:
                id_translate[obj,0] = obj+1
                id_translate[obj,1] = -1
        else:
            id_translate[obj,0] = obj+1
            id_translate[obj,1] = -1

    # adjust the directory strucutre accordingly
    obj_splitmerge_clean = {}

    if obj_splitmerge != None:
        id_translate = id_translate.astype(int)  
        keys = np.copy(list(obj_splitmerge.keys()))
        for jj in range(len(keys)):
            obj_loc = np.where(int(list(keys)[jj]) == id_translate[:,0])[0][0]
            if id_translate[obj_loc,1] == -1:
                del obj_splitmerge[list(keys)[jj]]

        # loop over objects and relable their indices if nescessary
        obj_splitmerge_clean = {}
        keys = np.copy(list(obj_splitmerge.keys()))
        core_translate = np.isin(id_translate[:,0], keys.astype(int))
        id_translate = id_translate[core_translate,:]
        for jj in range(len(keys)):
            obj_loc = np.where(int(list(keys)[jj]) == id_translate[:,0])[0][0]
            mergsplit = np.array(obj_splitmerge[keys[jj]])
            for kk in range(id_translate.shape[0]):
                mergsplit[np.isin(mergsplit, id_translate[kk,0])] = id_translate[kk,1]
            obj_splitmerge_clean[str(int(id_translate[obj_loc,1]))] = mergsplit
        
    return objectsTMP, obj_splitmerge_clean


def overlapping_objects(Object1,
                     Object2,
                     Data_to_mask):
    """ Function that finds all Objects1 that overlap with 
        objects in Object2
    """

    obj_structure_2D = np.zeros((3,3,3))
    obj_structure_2D[1,:,:] = 1
    objects_id_1, num_objects1 = ndimage.label(Object1.astype('int'), structure=obj_structure_2D)
    object_indices = ndimage.find_objects(objects_id_1)

    MaskedData = np.copy(Object1)
    MaskedData[:] = 0
    for obj in range(len(object_indices)):
        if object_indices[obj] != None:
            Obj1_tmp = Object1[object_indices[obj]]
            Obj2_tmp = Object2[object_indices[obj]]
            if np.sum(Obj1_tmp[Obj2_tmp > 0]) > 0:
                MaskedDataTMP = Data_to_mask[object_indices[obj]]
                MaskedDataTMP[Obj1_tmp == 0] = 0
                MaskedData[object_indices[obj]] = MaskedDataTMP
            
    return MaskedData


def smooth_uniform(
            data,       # matrix to smooth [time, lat, lon]
            t_smoot,    # temporal window length [time steps]
            xy_smooth,  # spatial window [grid cells]
            ):
    '''
    Function to spatiotemporal smooth atmospheric fiels even 
    if they contain missing data
    '''
    if np.isnan(data).any() == False:
        smooth_data = ndimage.uniform_filter(data, 
                                      size=[int(t_smoot),
                                            int(xy_smooth),
                                            int(xy_smooth)])
    else:
        # smoothing with missing values
        U = data.copy()
        V = data.copy()
        V[np.isnan(U)] = 0
        VV = ndimage.uniform_filter(V, size=[int(t_smoot),
                                            int(xy_smooth),
                                            int(xy_smooth)])
        W = 0*U.copy()+1
        W[np.isnan(U)] = 0
        WW = ndimage.uniform_filter(W, size=[int(t_smoot),
                                            int(xy_smooth),
                                            int(xy_smooth)])
        smooth_data = VV/WW
    return smooth_data



def jetstream_tracking(
                      uv200,            # wind speed at 200 hPa [m/s] - [time,lat,lon]
                      js_min_anomaly,   # minimum anomaly to indeitify jet objects [m/s]
                      MinTimeJS,        # minimum lifetime of jet objects [h]
                      dT,               # data time step [h]
                      Gridspacing,
                      connectLon
                      ):
    
    '''
    function to track jetstream objects
    '''

    uv200_smooth = smooth_uniform(uv200,
                             1,
                             int(500/(Gridspacing/1000.)))
    uv200smoothAn = smooth_uniform(uv200,
                                 int(78/dT),
                                 int(int(5000/(Gridspacing/1000.))))

    uv200_Anomaly = uv200_smooth - uv200smoothAn
    jet = uv200_Anomaly[:,:,:] >= js_min_anomaly


    #     Pressure_anomaly[np.isnan(Pressure_anomaly)] = 0
    #     jet[:,Mask == 0] = 0
    rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1
    rgiObjectsUD, nr_objectsUD = ndimage.label(jet, structure=rgiObj_Struct)
    print('            '+str(nr_objectsUD)+' object found')

    jet_objects, _ = clean_up_objects(rgiObjectsUD,
                                min_tsteps=int(MinTimeJS/dT),
                                 dT = dT)

    print('        break up long living CY objects that heve many elements')
    jet_objects, object_split = BreakupObjects(jet_objects,
                                int(MinTimeJS/dT),
                                dT)

#     jet_objects, object_split = clean_up_objects(rgiObjectsUD,
#                                 min_tsteps=int(MinTimeJS/dT),
#                                 dT = dT,
#                                 obj_splitmerge = object_split)
    
    if connectLon == 1:
        print('        connect cyclones objects over date line')
        jet_objects = ConnectLon_on_timestep(jet_objects)

    return jet_objects, object_split




def ar_850hpa_tracking(
                    VapTrans,        # 850 hPa moisture flux [g/g m/s] - [time,lat,lon]
                    MinMSthreshold,
                    MinTimeMS,
                    MinAreaMS,
                    Area,
                    dT,
                    connectLon
                ):
    
    '''
    function to track mosture streams and ARS according to 850 hPa moisture flux
    '''

    potARs = (VapTrans > MinMSthreshold)
    rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1
    rgiObjectsAR, nr_objectsUD = ndimage.label(potARs, structure=rgiObj_Struct)
    print('            '+str(nr_objectsUD)+' object found')

    # sort the objects according to their size
    Objects=ndimage.find_objects(rgiObjectsAR)

    rgiAreaObj = np.array([[np.sum(Area[Objects[ob][1:]][rgiObjectsAR[Objects[ob]][tt,:,:] == ob+1]) for tt in range(rgiObjectsAR[Objects[ob]].shape[0])] for ob in range(nr_objectsUD)])

    # create final object array
    MS_objectsTMP=np.copy(rgiObjectsAR); MS_objectsTMP[:]=0
    ii = 1
    for ob in range(len(rgiAreaObj)):
        AreaTest = np.max(np.convolve(np.array(rgiAreaObj[ob]) >= MinAreaMS*1000**2, np.ones(int(MinTimeMS/dT)), mode='valid'))
        if (AreaTest == int(MinTimeMS/dT)) & (len(rgiAreaObj[ob]) >= int(MinTimeMS/dT)):
            MS_objectsTMP[rgiObjectsAR == (ob+1)] = ii
            ii = ii + 1
    # lable the objects from 1 to N
    # MS_objects=np.copy(rgiObjectsAR); MS_objects[:]=0
    # Unique = np.unique(MS_objectsTMP)[1:]
    # ii = 1
    # for ob in range(len(Unique)):
    #     MS_objects[MS_objectsTMP == Unique[ob]] = ii
    #     ii = ii + 1

    MS_objects, _ = clean_up_objects(MS_objectsTMP,
                                  dT,
                                  min_tsteps=0)

    print('        break up long living MS objects that have many elements')
    MS_objects, object_split = BreakupObjects(MS_objects,
                                int(MinTimeMS/dT),
                                dT)

    if connectLon == 1:
        print('        connect MS objects over date line')
        MS_objects = ConnectLon_on_timestep(MS_objects)

    return MS_objects



def ar_ivt_tracking(IVT,
                    IVTtrheshold,
                    MinTimeIVT,
                    dT,
                    connectLon):
            
            potIVTs = (IVT > IVTtrheshold)
            rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1
            rgiObjectsIVT, nr_objectsUD = ndimage.label(potIVTs, structure=rgiObj_Struct)
            print('        '+str(nr_objectsUD)+' object found')

            IVT_objects, _ = clean_up_objects(rgiObjectsIVT,
                                           dT,
                                    min_tsteps=int(MinTimeIVT/dT))

            print('        break up long living IVT objects that have many elements')
            IVT_objects, object_split = BreakupObjects(IVT_objects,
                                         int(MinTimeIVT/dT),
                                        dT)

            if connectLon == 1:
                print('        connect IVT objects over date line')
                IVT_objects = ConnectLon_on_timestep(IVT_objects)

            return IVT_objects
        

        
def ar_check(objects_mask,
             AR_Lat,
             AR_width_lenght_ratio,
             AR_MinLen,
             Lon,
             Lat):

    start = time.perf_counter()
    AR_obj = np.copy(objects_mask); AR_obj[:] = 0.
    Objects=ndimage.find_objects(objects_mask.astype(int))

    aa=1
    for ii in range(len(Objects)):
        if Objects[ii] == None:
            continue
        ObjACT = objects_mask[Objects[ii]] == ii+1
        LonObj = np.array(Lon[Objects[ii][1],Objects[ii][2]])
        LatObj = np.array(Lat[Objects[ii][1],Objects[ii][2]])
        # check if object crosses the date line
        if LonObj.max()-LonObj.min() > 359:
            ObjACT = np.roll(ObjACT, int(ObjACT.shape[2]/2), axis=2)

        OBJ_max_len = np.zeros((ObjACT.shape[0]))
        for tt in range(ObjACT.shape[0]):
            PointsObj = np.append(LonObj[ObjACT[tt,:,:]==1][:,None], LatObj[ObjACT[tt,:,:]==1][:,None], axis=1)
            try:
                Hull = scipy.spatial.ConvexHull(np.array(PointsObj))
            except:
                continue
            XX = []; YY=[]
            for simplex in Hull.simplices:
    #                 plt.plot(PointsObj[simplex, 0], PointsObj[simplex, 1], 'k-')
                XX = XX + [PointsObj[simplex, 0][0]] 
                YY = YY + [PointsObj[simplex, 1][0]]

            points = [[XX[ii],YY[ii]] for ii in range(len(YY))]
            BOX = minimum_bounding_rectangle(np.array(PointsObj))

            DIST = np.zeros((3))
            for rr in range(3):
                DIST[rr] = DistanceCoord(BOX[rr][0],BOX[rr][1],BOX[rr+1][0],BOX[rr+1][1])
            OBJ_max_len[tt] = np.max(DIST)
            if OBJ_max_len[tt] <= AR_MinLen:
                ObjACT[tt,:,:] = 0
            else:
                rgiCenter = np.round(ndimage.measurements.center_of_mass(ObjACT[tt,:,:])).astype(int)
                LatCent = LatObj[rgiCenter[0],rgiCenter[1]]
                if np.abs(LatCent) < AR_Lat:
                    ObjACT[tt,:,:] = 0
            # check width to lenght ratio
            if DIST.max()/DIST.min() < AR_width_lenght_ratio:
                ObjACT[tt,:,:] = 0
        if LonObj.max()-LonObj.min() > 359:
            ObjACT = np.roll(ObjACT, -int(ObjACT.shape[2]/2), axis=2)
        ObjACT = ObjACT.astype(int)
        ObjACT[ObjACT!=0] = aa
        ObjACT = ObjACT + AR_obj[Objects[ii]]
        AR_obj[Objects[ii]] = ObjACT
        aa=aa+1

    return AR_obj

    end = time.perf_counter()
    timer(start, end)
    

    
def frontal_identification(Frontal_Diagnostic,
                                  front_treshold,
                                  MinAreaFR,
                                  Area):
            
            rgiObj_Struct_Fronts=np.zeros((3,3,3)); rgiObj_Struct_Fronts[1,:,:]=1
            Fmask = (Frontal_Diagnostic > front_treshold)

            rgiObjectsUD, nr_objectsUD = ndimage.label(Fmask,structure=rgiObj_Struct_Fronts)
            print('        '+str(nr_objectsUD)+' object found')

            # # calculate object size
            Objects=ndimage.find_objects(rgiObjectsUD)
            rgiAreaObj = np.array([np.sum(Area[Objects[ob][1:]][rgiObjectsUD[Objects[ob]][0,:,:] == ob+1]) for ob in range(nr_objectsUD)])

            # rgiAreaObj=np.array([np.sum(rgiObjectsUD[Objects[ob]] == ob+1) for ob in range(nr_objectsUD)])
            # create final object array
            FR_objects=np.copy(rgiObjectsUD)
            TooSmall = np.where(rgiAreaObj < MinAreaFR*1000**2)
            FR_objects[np.isin(FR_objects, TooSmall[0]+1)] = 0
            
            return FR_objects
        
        
        
def cy_acy_psl_tracking(
                    slp,
                    MaxPresAnCY,
                    MinTimeCY,
                    MinPresAnACY,
                    MinTimeACY,
                    dT,
                    Gridspacing,
                    connectLon
                    ):

    print('        track cyclones')
    rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1

    slp = slp/100.
    slp_smooth = smooth_uniform(slp,
                               1,
                               int(100/(Gridspacing/1000.)))
    slpsmoothAn = smooth_uniform(slp,
                                int(78/dT),
                                int(int(3000/(Gridspacing/1000.))))

    slp_Anomaly = slp_smooth-slpsmoothAn
#     slp_Anomaly[:,Mask == 0] = np.nan
    # plt.contour(slp_Anomaly[tt,:,:], levels=[-9990,-10,1100], colors='b')
    Pressure_anomaly = slp_Anomaly < MaxPresAnCY # 10 hPa depression | original setting was 12

    rgiObjectsUD, nr_objectsUD = ndimage.label(Pressure_anomaly, structure=rgiObj_Struct)
    print('            '+str(nr_objectsUD)+' object found')

    CY_objects, _ = clean_up_objects(rgiObjectsUD,
                          dT,
                    min_tsteps=int(MinTimeCY/dT))

    print('            break up long living CY objects that heve many elements')
    CY_objects, object_split = BreakupObjects(CY_objects,
                                int(MinTimeCY/dT),
                                dT)
    if connectLon == 1:
        print('            connect cyclones objects over date line')
        CY_objects = ConnectLon_on_timestep(CY_objects)
        
    
    
    print('        track anti-cyclones')
    HighPressure_annomaly = slp_Anomaly > MinPresAnACY # 12
    rgiObjectsUD, nr_objectsUD = ndimage.label(HighPressure_annomaly,structure=rgiObj_Struct)
    print('        '+str(nr_objectsUD)+' object found')
    
    ACY_objects, _ = clean_up_objects(rgiObjectsUD,
                                   dT,
                            min_tsteps=int(MinTimeACY/dT))

    print('            break up long living ACY objects that have many elements')
    ACY_objects, object_split = BreakupObjects(ACY_objects,
                                int(MinTimeCY/dT),
                                dT)
    if connectLon == 1:
        # connect objects over date line
        ACY_objects = ConnectLon_on_timestep(ACY_objects)
    
    return CY_objects, ACY_objects



def cy_acy_z500_tracking(
                    z500,
                    MinTimeCY,
                    dT,
                    Gridspacing,
                    connectLon
                    ):

    rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1
    z500 = z500 / 9.81
    z500_smooth = smooth_uniform(z500,
                                1,
                                int(100/(Gridspacing/1000.)))
    z500smoothAn = smooth_uniform(z500,
                                int(78/dT),
                                int(int(3000/(Gridspacing/1000.))))
    z500_Anomaly = z500_smooth - z500smoothAn
#     z500_Anomaly[:,Mask == 0] = np.nan

    z_low = z500_Anomaly < -80
    z_high = z500_Anomaly > 70

    # -------------------------------------
    print('    track 500 hPa cyclones')
#             z_low[:,Mask == 0] = 0
    rgiObjectsUD, nr_objectsUD = ndimage.label(z_low, structure=rgiObj_Struct)
    print('            '+str(nr_objectsUD)+' object found')

    cy_z500_objects, _ = clean_up_objects(rgiObjectsUD,
                                min_tsteps=int(MinTimeCY/dT),
                                 dT = dT)


    print('        break up long living CY objects that heve many elements')
    cy_z500_objects, object_split = BreakupObjects(cy_z500_objects,
                                int(MinTimeCY/dT),
                                dT)
    if connectLon == 1:
        print('        connect cyclones objects over date line')
        cy_z500_objects = ConnectLon_on_timestep(cy_z500_objects)


    # -------------------------------------
    print('    track 500 hPa anticyclones')
    rgiObjectsUD, nr_objectsUD = ndimage.label(z_high, structure=rgiObj_Struct)
    print('            '+str(nr_objectsUD)+' object found')
    acy_z500_objects, _ = clean_up_objects(rgiObjectsUD,
                                min_tsteps=int(MinTimeCY/dT),
                                 dT = dT)

    print('        break up long living CY objects that heve many elements')
    acy_z500_objects, object_split = BreakupObjects(acy_z500_objects,
                                int(MinTimeCY/dT),
                                dT)
    if connectLon == 1:
        print('        connect cyclones objects over date line')
        acy_z500_objects = ConnectLon_on_timestep(acy_z500_objects)

    return cy_z500_objects, acy_z500_objects






def col_identification(cy_z500_objects,
                       z500,
                       u200,
                       Frontal_Diagnostic,
                       MinTimeC,
                       dx,
                       dy,
                       Lon,
                       Lat
                      ):

    # area arround cyclone
    col_buffer = 500000 # m

    # check if cyclone is COL
    Objects=ndimage.find_objects(cy_z500_objects.astype(int))
    col_obj = np.copy(cy_z500_objects); col_obj[:]=0
    for ii in range(len(Objects)):
        if Objects[ii] == None:
            continue
        ObjACT = cy_z500_objects[Objects[ii]] == ii+1
        if ObjACT.shape[0] < MinTimeC:
            continue

        dxObj = abs(np.mean(dx[Objects[ii][1],Objects[ii][2]]))
        dyObj = abs(np.mean(dy[Objects[ii][1],Objects[ii][2]]))
        col_buffer_obj_lo = int(col_buffer/dxObj)
        col_buffer_obj_la = int(col_buffer/dyObj)

        # add buffer to object slice
        tt_start = Objects[ii][0].start
        tt_stop = Objects[ii][0].stop
        lo_start = Objects[ii][2].start - col_buffer_obj_lo 
        lo_stop = Objects[ii][2].stop + col_buffer_obj_lo
        la_start = Objects[ii][1].start - col_buffer_obj_la 
        la_stop = Objects[ii][1].stop + col_buffer_obj_la
        if lo_start < 0:
            lo_start = 0
        if lo_stop >= Lon.shape[1]:
            lo_stop = Lon.shape[1]-1
        if la_start < 0:
            la_start = 0
        if la_stop >= Lon.shape[0]:
            la_stop = Lon.shape[0]-1

        LonObj = Lon[la_start:la_stop, lo_start:lo_stop]
        LatObj = Lat[la_start:la_stop, lo_start:lo_stop]

        z500_ACT = np.copy(z500[tt_start:tt_stop, la_start:la_stop, lo_start:lo_stop])
        ObjACT = cy_z500_objects[tt_start:tt_stop, la_start:la_stop, lo_start:lo_stop] == ii+1
        u200_ob = u200[tt_start:tt_stop, la_start:la_stop, lo_start:lo_stop]
        front_ob = Frontal_Diagnostic[tt_start:tt_stop, la_start:la_stop, lo_start:lo_stop]
        if LonObj[0,-1] - LonObj[0,0] > 358:
            sift_lo = 'yes'
            # object crosses the date line
            shift = int(LonObj.shape[1]/2)
            LonObj = np.roll(LonObj, shift, axis=1)
            LatObj = np.roll(LatObj, shift, axis=1)
            z500_ACT = np.roll(z500_ACT, shift, axis=2)
            ObjACT = np.roll(ObjACT, shift, axis=2)
            u200_ob = np.roll(u200_ob, shift, axis=2)
            front_ob = np.roll(front_ob, shift, axis=2)
        else:
            sift_lo = 'no'

        # find location of z500 minimum
        z500_ACT_obj = np.copy(z500_ACT)
        z500_ACT_obj[ObjACT == 0] = 999999999999.

        for tt in range(z500_ACT_obj.shape[0]):
            min_loc = np.where(z500_ACT_obj[tt,:,:] == np.nanmin(z500_ACT_obj[tt]))
            min_la = min_loc[0][0]
            min_lo = min_loc[1][0]
            la_0 = min_la - col_buffer_obj_la
            if la_0 < 0:
                la_0 = 0
            lo_0 = min_lo - col_buffer_obj_lo
            if lo_0 < 0:
                lo_0 = 0

            lat_reg = LatObj[la_0:min_la + col_buffer_obj_la+1,
                             lo_0:min_lo + col_buffer_obj_lo+1]
            lon_reg = LonObj[la_0:min_la + col_buffer_obj_la+1,
                             lo_0:min_lo + col_buffer_obj_lo+1]

            col_region = z500_ACT[tt,
                                  la_0:min_la + col_buffer_obj_la+1,
                                  lo_0:min_lo + col_buffer_obj_lo+1]
            obj_col_region = z500_ACT_obj[tt,
                                  la_0:min_la + col_buffer_obj_la+1,
                                  lo_0:min_lo + col_buffer_obj_lo+1]
            min_z500_obj = z500_ACT[tt,min_la,min_lo]
            u200_ob_region = u200_ob[tt,
                                  la_0:min_la + col_buffer_obj_la+1,
                                  lo_0:min_lo + col_buffer_obj_lo+1]
            front_ob_region = front_ob[tt,
                                  la_0:min_la + col_buffer_obj_la+1,
                                  lo_0:min_lo + col_buffer_obj_lo+1]


            # check if 350 km radius arround center has higher Z
            min_loc_tt = np.where(obj_col_region[:,:] == 
                                  np.nanmin(z500_ACT_obj[tt]))
            min_la_tt = min_loc_tt[0][0]
            min_lo_tt = min_loc_tt[1][0]

            rdist = radialdistance(lat_reg[min_la_tt,min_lo_tt],
                                   lon_reg[min_la_tt,min_lo_tt],
                                   lat_reg,
                                   lon_reg)

            # COL should only occure between 20 and 70 degrees
            # https://journals.ametsoc.org/view/journals/clim/33/6/jcli-d-19-0497.1.xml
            if (abs(lat_reg[min_la_tt,min_lo_tt]) < 20) | (abs(lat_reg[min_la_tt,min_lo_tt]) > 70):
                ObjACT[tt,:,:] = 0
                continue

            # remove cyclones that are close to the poles
            if np.max(np.abs(lat_reg)) > 88:
                ObjACT[tt,:,:] = 0
                continue

            if np.nanmin(z500_ACT_obj[tt]) > 100000:
                # there is no object to process
                ObjACT[tt,:,:] = 0
                continue

            # CRITERIA 1) at least 75 % of grid cells in ring have have 10 m higher Z than center
            ring = (rdist >= (350 - (dxObj/1000.)*2))  & (rdist <= (350 + (dxObj/1000.)*2))
            if np.sum((min_z500_obj - col_region[ring]) < -10) < np.sum(ring)*0.75:
                ObjACT[tt,:,:] = 0
                continue

            # CRITERIA 2) check if 200 hPa wind speed is eastward in the poleward direction of the cyclone
            if lat_reg[min_la_tt,min_lo_tt] > 0:
                east_flow = u200_ob_region[0 : min_la_tt,
                                    min_lo_tt]
            else:
                east_flow = u200_ob_region[min_la_tt : -1,
                                    min_lo_tt]

            try:
                if np.min(east_flow) > 0:
                    ObjACT[tt,:,:] = 0
                    continue
            except:
                ObjACT[tt,:,:] = 0
                continue

            # Criteria 3) frontal zone in eastern flank of COL
            front_test = np.sum(np.abs(front_ob_region[:, min_lo_tt:]) > 1)
            if front_test < 1:
                ObjACT[tt,:,:] = 0
                continue

        if sift_lo == 'yes':
            ObjACT = np.roll(ObjACT, -shift, axis=2)

        ObjACT = ObjACT.astype('int')
        ObjACT[ObjACT > 0] = ii+1
        ObjACT = ObjACT + col_obj[tt_start:tt_stop, la_start:la_stop, lo_start:lo_stop]
        col_obj[tt_start:tt_stop, la_start:la_stop, lo_start:lo_stop] = ObjACT

    return col_obj





def mcs_tb_tracking(
                    tb,
                    pr,
                    SmoothSigmaC,
                    Pthreshold,
                    CL_Area,
                    CL_MaxT,
                    Cthreshold,
                    MinAreaC,
                    MinTimeC,
                    MCS_minPR,
                    MCS_minTime,
                    MCS_Minsize,
                    dT,
                    Area,
                    connectLon,
                   ):

    print('        track  clouds')
    rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1
    Csmooth=gaussian_filter(tb, sigma=(0,SmoothSigmaC,SmoothSigmaC))
    Cmask = (Csmooth <= Cthreshold)
    rgiObjectsC, nr_objectsUD = ndimage.label(Cmask, structure=rgiObj_Struct)
    print('        '+str(nr_objectsUD)+' cloud object found')

    if connectLon == 1:
        # connect objects over date line
        rgiObjectsC = ConnectLon(rgiObjectsC)
        
    # minimum cloud volume
    Objects=ndimage.find_objects(rgiObjectsC)
    
    rgiAreaObj = np.array([[np.sum(Area[Objects[ob][1],Objects[ob][2]][rgiObjectsC[Objects[ob]][tt,:,:] == ob+1]) for tt in range(rgiObjectsC[Objects[ob]].shape[0])] for ob in range(nr_objectsUD)])

    # rgiVolObjC=np.array([np.sum(rgiObjectsC[Objects[ob]] == ob+1) for ob in range(nr_objectsUD)])

    # create final object array
    C_objects=np.copy(rgiObjectsC); C_objects[:]=0
    ii = 1
    for ob in range(len(rgiAreaObj)):
        AreaTest = np.max(np.convolve(np.array(rgiAreaObj[ob]) >= MinAreaC*1000**2, np.ones(int(MinTimeC/dT)), mode='valid'))
        if (AreaTest == int(MinTimeC/dT)) & (len(rgiAreaObj[ob]) >=int(MinTimeC/dT)):
        # if rgiVolObjC[ob] >= MinAreaC:
            C_objects[rgiObjectsC == (ob+1)] = ii
            ii = ii + 1

    print('        break up long living cloud shield objects that have many elements')
    C_objects, object_split = BreakupObjects(C_objects,
                                int(MinTimeC/dT),
                                dT)
    if connectLon == 1:
        print('        connect cloud objects over date line')
        C_objects = ConnectLon_on_timestep(C_objects)



    # check if precipitation object is from an MCS
    object_indices = ndimage.find_objects(C_objects)
    MCS_objects_Tb = np.zeros(C_objects.shape,dtype=int)

    for iobj,_ in tqdm(enumerate(object_indices)):
        if object_indices[iobj] is None:
            continue

        time_slice = object_indices[iobj][0]
        lat_slice  = object_indices[iobj][1]
        lon_slice  = object_indices[iobj][2]

        tb_object_slice= C_objects[object_indices[iobj]]
        tb_object_act = np.where(tb_object_slice==iobj+1,True,False)
        if len(tb_object_act) < MCS_minTime:
            continue

        tb_slice =  tb[object_indices[iobj]]
        tb_act = np.copy(tb_slice)
        tb_act[~tb_object_act] = np.nan

        bt_object_slice = C_objects[object_indices[iobj]]
        bt_object_act = np.copy(bt_object_slice)
        bt_object_act[~tb_object_act] = 0

        area_act = np.tile(Area[lat_slice, lon_slice], (tb_act.shape[0], 1, 1))
        area_act[~tb_object_act] = 0

        ### Calculate cloud properties
        tb_size = np.array(np.sum(area_act,axis=(1,2)))
        tb_min = np.array(np.nanmin(tb_act,axis=(1,2)))

        ### Calculate precipitation properties
        pr_act = np.copy(pr[object_indices[iobj]])
        pr_act[tb_object_act == 0] = np.nan

        pr_peak_act = np.array(np.nanmax(pr_act,axis=(1,2)))

        pr_region_act = pr_act >= Pthreshold*dT
        area_act = np.tile(Area[lat_slice, lon_slice], (tb_act.shape[0], 1, 1))
        area_act[~pr_region_act] = 0
        pr_under_cloud = np.array(np.sum(area_act,axis=(1,2)))/1000**2 


        # Test if object classifies as MCS
        tb_size_test = np.max(np.convolve((tb_size / 1000**2 >= CL_Area), np.ones(MCS_minTime), 'valid') / MCS_minTime) == 1
        tb_overshoot_test = np.max((tb_min  <= CL_MaxT )) == 1
        pr_peak_test = np.max(np.convolve((pr_peak_act >= MCS_minPR ), np.ones(MCS_minTime), 'valid') / MCS_minTime) ==1
        pr_area_test = np.max((pr_under_cloud >= MCS_Minsize)) == 1
        MCS_test = (
                    tb_size_test
                    & tb_overshoot_test
                    & pr_peak_test
                    & pr_area_test
        )

        # assign unique object numbers
        tb_object_act = np.array(tb_object_act).astype(int)
        tb_object_act[tb_object_act == 1] = iobj + 1

#         window_length = int(MCS_minTime / dT)
#         moving_averages = np.convolve(MCS_test, np.ones(window_length), 'valid') / window_length

    #     if iobj+1 == 19:
    #         stop()
        if MCS_test == 1:
            TMP = np.copy(MCS_objects_Tb[object_indices[iobj]])
            TMP = TMP + tb_object_act
            MCS_objects_Tb[object_indices[iobj]] = TMP

        else:
            continue

    MCS_objects_Tb, _ = clean_up_objects(MCS_objects_Tb,
                                       dT,
                                       min_tsteps=int(MCS_minTime/dT))  

    return MCS_objects_Tb, C_objects



def tc_tracking(CY_objects,
                t850,
                slp,
                tb,
                C_objects,
                Lon,
                Lat,
                TC_lat_genesis,
                TC_deltaT_core,
                TC_T850min,
                TC_Pmin,
                TC_lat_max
               ):
    TC_Tracks = {}
    Objects=ndimage.find_objects(CY_objects.astype(int))
    TC_obj = np.copy(CY_objects); TC_obj[:]=0
    for ii in range(len(Objects)):
        if Objects[ii] == None:
            continue
        ObjACT = CY_objects[Objects[ii]] == ii+1
        if ObjACT.shape[0] < 2*8:
            continue
        T_ACT = np.copy(t850[Objects[ii]])
        slp_ACT = np.copy(slp[Objects[ii]])/100.
        LonObj = Lon[Objects[ii][1],Objects[ii][2]]
        LatObj = Lat[Objects[ii][1],Objects[ii][2]]
        # check if object crosses the date line
        if LonObj.max()-LonObj.min() > 359:
            ObjACT = np.roll(ObjACT, int(ObjACT.shape[2]/2), axis=2)
            slp_ACT = np.roll(slp_ACT, int(ObjACT.shape[2]/2), axis=2)
        # Calculate low pressure center track
        slp_ACT[ObjACT == 0] = 999999999.
        Track_ACT = np.array([np.argwhere(slp_ACT[tt,:,:] == np.nanmin(slp_ACT[tt,:,:]))[0] for tt in range(ObjACT.shape[0])])
        LatLonTrackAct = np.array([(LatObj[Track_ACT[tt][0],Track_ACT[tt][1]],LonObj[Track_ACT[tt][0],Track_ACT[tt][1]]) for tt in range(ObjACT.shape[0])])
        if np.min(np.abs(LatLonTrackAct[:,0])) > TC_lat_genesis:
            ObjACT[:] = 0
            continue
        else:

            # has the cyclone a warm core?
            DeltaTCore = np.zeros((ObjACT.shape[0])); DeltaTCore[:] = np.nan
            T850_core = np.copy(DeltaTCore)
            for tt in range(ObjACT.shape[0]):
                T_cent = np.mean(T_ACT[tt,Track_ACT[tt,0]-1:Track_ACT[tt,0]+2,Track_ACT[tt,1]-1:Track_ACT[tt,1]+2])
                T850_core[tt] = T_cent
                T_Cyclone = np.mean(T_ACT[tt,ObjACT[tt,:,:] != 0])
    #                     T_Cyclone = np.mean(T_ACT[tt,MassC[0]-5:MassC[0]+6,MassC[1]-5:MassC[1]+6])
                DeltaTCore[tt] = T_cent-T_Cyclone
            # smooth the data
            DeltaTCore = gaussian_filter(DeltaTCore,1)
            WarmCore = DeltaTCore > TC_deltaT_core

            if np.sum(WarmCore) < 8:
                continue
            ObjACT[WarmCore == 0,:,:] = 0
            # is the core temperature warm enough
            ObjACT[T850_core < TC_T850min,:,:] = 0


            # TC must have pressure of less 980 hPa
            MinPress = np.min(slp_ACT, axis=(1,2))
            if np.sum(MinPress < TC_Pmin) < 8:
                continue

            # # is the cloud shield cold enough?
            # BT_act = np.copy(tb[Objects[ii]])
            # # BT_objMean = np.zeros((BT_act.shape[0])); BT_objMean[:] = np.nan
            # # PR_objACT = np.copy(PR_objects[Objects[ii]])
            # # for tt in range(len(BT_objMean)):
            # #     try:
            # #         BT_objMean[tt] = np.nanmean(BT_act[tt,PR_objACT[tt,:,:] != 0])
            # #     except:
            # #         continue
            # BT_objMean = np.nanmean(BT_act[:,:,:], axis=(1,2))

            # # is cloud shild overlapping with TC?
            # BT_objACT = np.copy(C_objects[Objects[ii]])
            # bt_overlap = np.array([np.sum((BT_objACT[kk,ObjACT[kk,:,:] == True] > 0) == True)/np.sum(ObjACT[10,:,:] == True) for kk in range(ObjACT.shape[0])]) > 0.4

        # remove pieces of the track that are not TCs
        TCcheck = (T850_core > TC_T850min) & (WarmCore == 1) & (MinPress < TC_Pmin) #& (bt_overlap == 1) #(BT_objMean < TC_minBT)
        LatLonTrackAct[TCcheck == False,:] = np.nan

        Max_LAT = (np.abs(LatLonTrackAct[:,0]) >  TC_lat_max)
        LatLonTrackAct[Max_LAT,:] = np.nan

        if np.sum(~np.isnan(LatLonTrackAct[:,0])) == 0:
            continue

        # check if cyclone genesis is over water; each re-emergence of TC is a new genesis
        resultLAT = [list(map(float,g)) for k,g in groupby(LatLonTrackAct[:,0], np.isnan) if not k]
        resultLON = [list(map(float,g)) for k,g in groupby(LatLonTrackAct[:,1], np.isnan) if not k]
        LS_genesis = np.zeros((len(resultLAT))); LS_genesis[:] = np.nan
        for jj in range(len(resultLAT)):
            LS_genesis[jj] = is_land(resultLON[jj][0],resultLAT[jj][0])
        if np.max(LS_genesis) == 1:
            for jj in range(len(LS_genesis)):
                if LS_genesis[jj] == 1:
                    SetNAN = np.isin(LatLonTrackAct[:,0],resultLAT[jj])
                    LatLonTrackAct[SetNAN,:] = np.nan

        # make sure that only TC time slizes are considered
        ObjACT[np.isnan(LatLonTrackAct[:,0]),:,:] = 0

        if LonObj.max()-LonObj.min() > 359:
            ObjACT = np.roll(ObjACT, -int(ObjACT.shape[2]/2), axis=2)
        ObjACT = ObjACT.astype(int)
        ObjACT[ObjACT!=0] = ii+1

        ObjACT = ObjACT + TC_obj[Objects[ii]]
        TC_obj[Objects[ii]] = ObjACT
        TC_Tracks[str(ii+1)] = LatLonTrackAct

    return TC_obj, TC_Tracks



def mcs_pr_tracking(pr,
                    tb,
                    C_objects,
                    AR_obj,
                    Area,
                    Lon,
                    Lat,
                    SmoothSigmaP,
                    Pthreshold,
                    MinTimePR,
                    MCS_minPR,
                    MCS_Minsize,
                    CL_Area,
                    CL_MaxT,
                    MCS_minTime,
                    MinAreaPR,
                    dT,
                    connectLon):

    print('        track  precipitation')
    
    rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1
    PRsmooth=gaussian_filter(pr, sigma=(0,SmoothSigmaP,SmoothSigmaP))
    PRmask = (PRsmooth >= Pthreshold*dT)
    
    rgiObjectsPR, nr_objectsUD = ndimage.label(PRmask, structure=rgiObj_Struct)
    print('            '+str(nr_objectsUD)+' precipitation object found')

    if connectLon == 1:
        # connect objects over date line
        rgiObjectsPR = ConnectLon(rgiObjectsPR)

    stop()    
    
    # remove None objects
    Objects=ndimage.find_objects(rgiObjectsPR)
    rgiVolObj=np.array([np.sum(rgiObjectsPR[Objects[ob]] == ob+1) for ob in range(nr_objectsUD)])
    ZERO_V =  np.where(rgiVolObj == 0)
    if len(ZERO_V[0]) > 0:
        Dummy = (slice(0, 1, None), slice(0, 1, None), slice(0, 1, None))
        Objects = np.array(Objects)
        for jj in ZERO_V[0]:
            Objects[jj] = Dummy

    stop()
    # Remove objects that are too small or short lived
    rgiAreaObj = np.array([[np.sum(Area[Objects[ob][1],Objects[ob][2]][rgiObjectsPR[Objects[ob]][tt,:,:] == ob+1]) for tt in range(rgiObjectsPR[Objects[ob]].shape[0])] for ob in range(nr_objectsUD)])
    # create final object array
    PR_objects=np.copy(rgiObjectsPR); PR_objects[:]=0
    ii = 1
    for ob in range(len(rgiAreaObj)):
        AreaTest = np.max(np.convolve(np.array(rgiAreaObj[ob]) >= MinAreaPR*1000**2, np.ones(int(MinTimePR/dT)), mode='valid'))
        if (AreaTest == int(MinTimePR/dT)) & (len(rgiAreaObj[ob]) >= int(MinTimePR/dT)):
            PR_objects[rgiObjectsPR == (ob+1)] = ii
            ii = ii + 1

    print('            break up long living precipitation objects that have many elements')
    PR_objects, object_split = BreakupObjects(PR_objects,
                                int(MinTimePR/dT),
                                dT)

    if connectLon == 1:
        print('            connect precipitation objects over date line')
        PR_objects = ConnectLon_on_timestep(PR_objects)

    # ===================
    print('    check if pr objects quallify as MCS')

    Objects=ndimage.find_objects(PR_objects.astype(int))
    MCS_obj = np.copy(PR_objects); MCS_obj[:]=0
    window_length = int(MCS_minTime/dT)
    for ii in range(len(Objects)):
        if Objects[ii] == None:
            continue
        ObjACT = PR_objects[Objects[ii]] == ii+1
        if ObjACT.shape[0] < 2:
            continue
        if ObjACT.shape[0] < window_length:
            continue
        Cloud_ACT = np.copy(C_objects[Objects[ii]])
        LonObj = Lon[Objects[ii][1],Objects[ii][2]]
        LatObj = Lat[Objects[ii][1],Objects[ii][2]]   
        Area_ACT = Area[Objects[ii][1],Objects[ii][2]]
        PR_ACT = pr[Objects[ii]]

        PR_Size = np.array([np.sum(Area_ACT[ObjACT[tt,:,:] >0]) for tt in range(ObjACT.shape[0])])
        PR_MAX = np.array([np.max(PR_ACT[tt,ObjACT[tt,:,:] >0]) if len(PR_ACT[tt,ObjACT[tt,:,:]>0]) > 0 else 0 for tt in range(ObjACT.shape[0])])
        # Get cloud shield
        rgiCL_obj = np.delete(np.unique(Cloud_ACT[ObjACT > 0]),0)
        if len(rgiCL_obj) == 0:
            # no deep cloud shield is over the precipitation
            continue
        CL_OB_TMP = C_objects[Objects[ii][0]]
        CLOUD_obj_act = np.in1d(CL_OB_TMP.flatten(), rgiCL_obj).reshape(CL_OB_TMP.shape)
        Cloud_Size = np.array([np.sum(Area[CLOUD_obj_act[tt,:,:] >0]) for tt in range(CLOUD_obj_act.shape[0])])
        # min temperatur must be taken over precip area
    #     CL_ob_pr = C_objects[Objects[ii]]
        CL_BT_pr = np.copy(tb[Objects[ii]])
        CL_BT_pr[ObjACT == 0] = np.nan
        Cloud_MinT = np.nanmin(CL_BT_pr, axis=(1,2))
    #     Cloud_MinT = np.array([np.min(CL_BT_pr[tt,CL_ob_pr[tt,:,:] >0]) if len(CL_ob_pr[tt,CL_ob_pr[tt,:,:] >0]) > 0 else 0 for tt in range(CL_ob_pr.shape[0])])
        Cloud_MinT[Cloud_MinT < 150 ] = np.nan
        # is precipitation associated with AR?
        AR_ob = np.copy(AR_obj[Objects[ii]])
        AR_ob[:,LatObj < 25] = 0 # only consider ARs in mid- and hight latitudes
        AR_test = np.sum(AR_ob > 0, axis=(1,2))            

        MCS_max_residence = np.min([int(24/dT),ObjACT.shape[0]]) # MCS criterion must be meet within this time window
                               # or MCS is discontinued
        # minimum lifetime peak precipitation
        is_pr_peak_intense = np.convolve(
                                        PR_MAX >= MCS_minPR*dT, 
                                        np.ones(MCS_max_residence), 'same') >= 1
        # minimum precipitation area threshold
        is_pr_size = np.convolve(
                        (np.convolve((PR_Size / 1000**2 >= MCS_Minsize), np.ones(window_length), 'same') / window_length) == 1, 
                                        np.ones(MCS_max_residence), 'same') >= 1
        # Tb size and time threshold
        is_Tb_area = np.convolve(
                        (np.convolve((Cloud_Size / 1000**2 >= CL_Area), np.ones(window_length), 'same') / window_length) == 1, 
                                        np.ones(MCS_max_residence), 'same') >= 1
        # Tb overshoot
        is_Tb_overshoot = np.convolve(
                            Cloud_MinT  <= CL_MaxT, 
                            np.ones(MCS_max_residence), 'same') >= 1
        try:
            MCS_test = (
                        (is_pr_peak_intense == 1)
                        & (is_pr_size == 1)
                        & (is_Tb_area == 1)
                        & (is_Tb_overshoot == 1)
                )
            ObjACT[MCS_test == 0,:,:] = 0
        except:
            ObjACT[MCS_test == 0,:,:] = 0



        # assign unique object numbers
        ObjACT = np.array(ObjACT).astype(int)
        ObjACT[ObjACT == 1] = ii+1

    #         # remove all precip that is associated with ARs
    #         ObjACT[AR_test > 0] = 0

    #     # PR area defines MCS area and precipitation
    #     window_length = int(MCS_minTime/dT)
    #     cumulative_sum = np.cumsum(np.insert(MCS_TEST, 0, 0))
    #     moving_averages = (cumulative_sum[window_length:] - cumulative_sum[:-window_length]) / window_length
    #     if ob == 16:
    #         stop()
        if np.max(MCS_test) == 1:
            TMP = np.copy(MCS_obj[Objects[ii]])
            TMP = TMP + ObjACT
            MCS_obj[Objects[ii]] = TMP
        else:
            continue

    MCS_obj, _ = clean_up_objects(MCS_obj,
                                   dT,
                            min_tsteps=int(MCS_minTime/dT))  

    return PR_objects, MCS_obj


def track_tropwaves(pr,
                   Lat,
                   connectLon,
                   dT):
    """ Identifies and tracks four types of tropical waves from
        hourly precipitation data:
        Mixed Rossby Gravity Waves
        n>=0 Eastward Inertio Gravirt Wave
        Kelvin Waves
        and n>=1 Inertio Gravirt Wave
    """
    
    
    from Tracking_Functions import interpolate_numba
    from Tracking_Functions import KFfilter
    from Tracking_Functions import clean_up_objects
    from Tracking_Functions import BreakupObjects
    from Tracking_Functions import ConnectLon_on_timestep

    pr_eq = pr.copy()
    pr_eq[:,np.abs(Lat[:,0]) > 20] = 0
    
    pr_eq = interpolate_numba(np.array(pr_eq))
    tropical_waves = KFfilter(pr_eq,
                     int(24/dT))

    er = KFfilter.erfilter(tropical_waves, fmin=None, fmax=None, kmin=-10, kmax=-1, hmin=0, hmax=90, n=1) # had to set hmin from 8 to 0
    eig0 = KFfilter.eig0filter(tropical_waves)
    kelvin = KFfilter.kelvinfilter(tropical_waves)
    igw = KFfilter.igfilter(tropical_waves)
    mrg = KFfilter.mrgfilter(tropical_waves)

    er_mask = er > 0.05
    mrg_mask = mrg > 0.05
    igw_mask = igw > 0.2
    kelvin_mask = kelvin > 0.1
    eig0_mask = eig0 > 0.1
    wave_names = ['ER','MRG','IGW','Kelvin','Eig0']

    print('        track tropical waves')
    rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1
    for wa in range(5):
        print('            work on '+wave_names[wa])
        if wa == 0:
            wave = er_mask.copy()
        if wa == 1:
            wave = mrg_mask.copy()
        elif wa == 2:
            wave = igw_mask.copy()
        elif wa == 3:
            wave = kelvin_mask.copy()
        elif wa == 4:
            wave = eig0_mask.copy()
        rgiObjectsUD, nr_objectsUD = ndimage.label(wave, structure=rgiObj_Struct)
        print('                '+str(nr_objectsUD)+' object found')

        wave_objects, _ = clean_up_objects(rgiObjectsUD,
                              dT,
                              min_tsteps=int(48/dT))

        print('                break up long tropical waves that heve many elements')
        wave_objects, object_split = BreakupObjects(wave_objects,
                                    int(48/dT),
                                    dT)
        if connectLon == 1:
            print('                connect waves objects over date line')
            wave_objects = ConnectLon_on_timestep(wave_objects)

        if wa == 0:
            er_objects = wave_objects.copy()
        if wa == 1:
            mrg_objects = wave_objects.copy()
        if wa == 2:
            igw_objects = wave_objects.copy()
        if wa == 3:
            kelvin_objects = wave_objects.copy()
        if wa == 4:
            eig0_objects = wave_objects.copy()

    del wave
    del wave_objects
    del pr_eq

    return mrg_objects, igw_objects, kelvin_objects, eig0_objects, er_objects





def moaap(
    Lon,                           # 2D longitude grid centers
    Lat,                           # 2D latitude grid spacing
    Time,                          # datetime vector of data
    dT,                            # integer - temporal frequency of data [hour]
    Mask,                          # mask with dimensions [lat,lon] defining analysis region
    v850 = None, # 850 hPa zonal wind speed [m/s]
    u850 = None, # 850 hPa meridional wind speed [m/s]
    t850 = None, # 850 hPa air temperature [K]
    q850 = None, # 850 hPa mixing ratio [g/kg]
    slp = None,  # sea level pressure [Pa]
    ivte = None, # zonal integrated vapor transport [kg m-1 s-1]
    ivtn = None, # meridional integrated vapor transport [kg m-1 s-1]
    z500 = None, # geopotential height [gpm]
    v200 = None, # 200 hPa zonal wind speed [m/s]
    u200 = None, # 200 hPa meridional wind speed [m/s]
    pr   = None, # accumulated surface precipitation [mm/time]
    tb   = None, # brightness temperature [K]
    DataName = '',                 # name of the common grid
    OutputFolder='',               # string containing the output directory path. Default is local directory
    # minimum precip. obj.
    SmoothSigmaP = 0,              # Gaussion std for precipitation smoothing
    Pthreshold = 2,                # precipitation threshold [mm/h]
    MinTimePR = 4,                 # minimum lifetime of precip. features in hours
    MinAreaPR = 5000,              # minimum area of precipitation features [km2]
    # minimum Moisture Stream 
    MinTimeMS = 9,                 # minimum lifetime for moisture stream [hours]
    MinAreaMS = 100000,            # mimimum area of moisture stream [km2]
    MinMSthreshold = 0.13,         # treshold for moisture stream [g*m/g*s]
    # cyclone tracking
    MinTimeCY = 12,                # minimum livetime of cyclones [hours]
    MaxPresAnCY = -8,              # preshure thershold for cyclone anomaly [hPa]
    # anty cyclone tracking
    MinTimeACY = 12,               # minimum livetime of anticyclone [hours]
    MinPresAnACY = 6,              # preshure thershold for anti cyclone anomaly [hPa]
    # Frontal zones
    MinAreaFR = 50000,             # mimimum size of frontal zones [km2]
    front_treshold = 1,            # threshold for masking frontal zones
    # Cloud tracking setup
    SmoothSigmaC = 0,              # standard deviation of Gaussian filter for cloud tracking
    Cthreshold = 241,              # brightness temperature threshold for cloud tracking [K]
    MinTimeC = 4,                  # mimimum livetime of ice cloud shields [hours]
    MinAreaC = 40000,              # mimimum area of ice cloud shields [km2]
    # AR tracking
    IVTtrheshold = 500,            # Integrated water vapor transport threshold for AR detection [kg m-1 s-1]
    MinTimeIVT = 9,                # minimum livetime of ARs [hours]
    AR_MinLen = 2000,              # mimimum length of an AR [km]
    AR_Lat = 20,                   # AR centroids have to be poeward of this latitude
    AR_width_lenght_ratio = 2,     # mimimum length to width ratio of AR
    # TC detection
    TC_Pmin = 995,                 # mimimum pressure for TC detection [hPa]
    TC_lat_genesis = 35,           # maximum latitude for TC genesis [absolute degree latitude]
    TC_lat_max = 60,               # maximum latitude for TC existance [absolute degree latitude]
    TC_deltaT_core = 0,            # minimum degrees difference between TC core and surrounding [K]
    TC_T850min = 285,              # minimum temperature of TC core at 850hPa [K]
    TC_minBT = 241,                # minimum average cloud top brightness temperature [K]
    # MCs detection
    MCS_Minsize = 5000,            # minimum size of precipitation area [km2] 
    MCS_minPR = 15,                # minimum precipitation threshold [mm/h]
    CL_MaxT = 215,                 # minimum brightness temperature in ice shield [K]
    CL_Area = 40000,               # minimum cloud area size [km2]
    MCS_minTime = 4,               # minimum lifetime of MCS [hours]
    js_min_anomaly = 24,           # jet minimal anomaly [m/s]
    MinTimeJS = 24,                # minimum lifetime of jets [h]
    tropwave_minTime = 48          # minimum lifetime of tropical waves [h]
    ):
    
    # calculate grid spacing assuming regular lat/lon grid
    _,_,Area,Gridspacing = calc_grid_distance_area(Lon,Lat)
    Area[Area < 0] = 0
    
    EarthCircum = 40075000 #[m]
    Lat = np.array(Lat)
    Lon = np.array(Lon)
    dLat = np.copy(Lon); dLat[:] = EarthCircum/(360/(Lat[1,0]-Lat[0,0]))
    dLon = np.copy(Lon)
    for la in range(Lat.shape[0]):
        dLon[la,:] = EarthCircum/(360/(Lat[1,0]-Lat[0,0]))*np.cos(np.deg2rad(Lat[la,0]))
    dLat = np.abs(dLat)
    dLon = np.abs(dLon)
    
    StartDay = Time[0]
    SetupString = DataName+'_dt-'+str(dT)+'h_MOAAP-masks'
    NCfile = OutputFolder + str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+DataName+'_ObjectMasks_'+SetupString+'.nc'
    FrontMask = np.copy(Mask)
    FrontMask[np.abs(Lat) < 10] = 0

    # connect over date line?
    if (Lon[0,0] < -176) & (Lon[0,-1] > 176):
        connectLon= 1
    else:
        connectLon= 0

    ### print out which phenomenon can be investigated
    if slp is not None:
        slp_test = 'yes'
    else:
        slp_test = 'no'
    if (ivte is not None) & (ivtn is not None):
        ar_test = 'yes'
    else:
        ar_test = 'no'
    if (v850 is not None) & (u850 is not None) & (t850 is not None):
        front_test = 'yes'
    else:
        front_test = 'no'
    if (slp is not None) & (tb is not None) \
       & (t850 is not None) & (pr is not None):
        tc_test = 'yes'
    else:
        tc_test = 'no'
    if z500 is not None:
        z500_test = 'yes'
    else:
        z500_test = 'no'
    if (z500 is not None) & (front_test == 'yes') & \
       (u200 is not None):
        col_test = 'yes'
    else:
        col_test = 'no'
    if (v200 is not None) & (u200 is not None):
        jet_test = 'yes'
    else:
        jet_test = 'no'
    if (pr is not None) & (tb is not None):
        mcs_tb_test = 'yes'
    else:
        mcs_tb_test = 'no'
    if (q850 is not None) & (v850 is not None) & \
       (u850 is not None):
        ms_test = 'yes'
    else:
        ms_test = 'no'
    if (pr is not None):
        ew_test = 'yes'
    else:
        ew_test = 'no'
    
    print(' ')
    print('The provided variables allow tracking the following phenomena')
    print(' ')
    print('|  phenomenon  | tracking |')
    print('---------------------------')
    print('   Jetstream   |   ' + jet_test)
    print('   PSL CY/ACY  |   ' + slp_test)
    print('   Z500 CY/ACY |   ' + z500_test)
    print('   COLs        |   ' + col_test)
    print('   IVT ARs     |   ' + ar_test)
    print('   MS ARs      |   ' + ms_test)
    print('   Fronts      |   ' + front_test)
    print('   TCs         |   ' + tc_test)
    print('   MCSs        |   ' + mcs_tb_test)
    print('   Equ. Waves  |   ' + ew_test)
    print('---------------------------')
    print(' ')
    
    import time
    
    # Mask data outside of Focus domain
    try:
        v850[:,Mask == 0] = np.nan
    except:
        pass
    try:
        u850[:,Mask == 0] = np.nan
    except:
        pass
    try:
        t850[:,Mask == 0] = np.nan
    except:
        pass
    try:
        q850[:,Mask == 0] = np.nan
    except:
        pass
    try:
        slp[:,Mask == 0]  = np.nan
    except:
        pass
    try:
        ivte[:,Mask == 0] = np.nan
    except:
        pass
    try:
        ivtn[:,Mask == 0] = np.nan
    except:
        pass
    try:
        z500[:,Mask == 0] = np.nan
    except:
        pass
    try:
        v200[:,Mask == 0] = np.nan
    except:
        pass
    try:
        u200[:,Mask == 0] = np.nan
    except:
        pass
    try:
        pr[:,Mask == 0]   = np.nan
    except:
        pass
    try:
        tb[:,Mask == 0]   = np.nan
    except:
        pass

    if jet_test == 'yes':
        print('======> track jetstream')
        start = time.perf_counter()
        uv200 = (u200 ** 2 + v200 ** 2) ** 0.5

        jet_objects, object_split = jetstream_tracking(uv200,
                                      js_min_anomaly,
                                      MinTimeJS,
                                      dT,
                                      Gridspacing,
                                      connectLon)
        jet_objects_characteristics = calc_object_characteristics(jet_objects, # feature object file
                                     uv200,         # original file used for feature detection
                                     OutputFolder+'jet_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                                     Time,            # timesteps of the data
                                     Lat,             # 2D latidudes
                                     Lon,             # 2D Longitudes
                                     Gridspacing,
                                     Area,
                                     min_tsteps=int(MinTimeJS/dT),
                                     split_merge = object_split)
        
        end = time.perf_counter()
        timer(start, end)
        
        
    if ew_test == 'yes':
        print('======> track tropical waves')
        start = time.perf_counter()
        mrg_objects, igw_objects, kelvin_objects, eig0_objects, er_objects = track_tropwaves(
                        pr,
                        Lat,
                        connectLon,
                        dT
                        )
        end = time.perf_counter()
        timer(start, end)
        
        gr_mrg = calc_object_characteristics(mrg_objects, # feature object file
                                 pr,         # original file used for feature detection
                                 OutputFolder+'MRG_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,        # output file name and locaiton
                                 Time,            # timesteps of the data
                                 Lat,             # 2D latidudes
                                 Lon,             # 2D Longitudes
                                 Gridspacing,
                                 Area,
                                 min_tsteps=int(tropwave_minTime/dT))      # minimum livetime in hours
        
        gr_igw = calc_object_characteristics(igw_objects, # feature object file
                                 pr,         # original file used for feature detection
                                 OutputFolder+'IGW_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,        # output file name and locaiton
                                 Time,            # timesteps of the data
                                 Lat,             # 2D latidudes
                                 Lon,             # 2D Longitudes
                                 Gridspacing,
                                 Area,
                                 min_tsteps=int(48/dT))      # minimum livetime in hours
        
        gr_kelvin = calc_object_characteristics(kelvin_objects, # feature object file
                                 pr,         # original file used for feature detection
                                 OutputFolder+'Kelvin_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,        # output file name and locaiton
                                 Time,            # timesteps of the data
                                 Lat,             # 2D latidudes
                                 Lon,             # 2D Longitudes
                                 Gridspacing,
                                 Area,
                                 min_tsteps=int(48/dT))      # minimum livetime in hours
        
        gr_eig0 = calc_object_characteristics(eig0_objects, # feature object file
                                 pr,         # original file used for feature detection
                                 OutputFolder+'Eig0_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,        # output file name and locaiton
                                 Time,            # timesteps of the data
                                 Lat,             # 2D latidudes
                                 Lon,             # 2D Longitudes
                                 Gridspacing,
                                 Area,
                                 min_tsteps=int(48/dT))      # minimum livetime in hours
        
        gr_er = calc_object_characteristics(er_objects, # feature object file
                                 pr,         # original file used for feature detection
                                 OutputFolder+'ER_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,        # output file name and locaiton
                                 Time,            # timesteps of the data
                                 Lat,             # 2D latidudes
                                 Lon,             # 2D Longitudes
                                 Gridspacing,
                                 Area,
                                 min_tsteps=int(48/dT))      # minimum livetime in hours
        
        
    if ms_test == 'yes':
        print('======> track moisture streams and atmospheric rivers (ARs)')
        start = time.perf_counter()
        VapTrans = ((u850 * q850)**2 + 
                    (v850 * q850)**2)**(1/2)
        
        MS_objects = ar_850hpa_tracking(
                                        VapTrans,
                                        MinMSthreshold,
                                        MinTimeMS,
                                        MinAreaMS,
                                        Area,
                                        dT,
                                        connectLon)
        
        grMSs = calc_object_characteristics(MS_objects, # feature object file
                                 VapTrans,         # original file used for feature detection
                                 OutputFolder+'MS850_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,        # output file name and locaiton
                                 Time,            # timesteps of the data
                                 Lat,             # 2D latidudes
                                 Lon,             # 2D Longitudes
                                 Gridspacing,
                                 Area,
                                 min_tsteps=int(MinTimeMS/dT))      # minimum livetime in hours
        
        end = time.perf_counter()
        timer(start, end)
        
    
    if ar_test == 'yes':
        print('======> track IVT streams and atmospheric rivers (ARs)')
        start = time.perf_counter()
        IVT = (ivte ** 2 + ivtn ** 2) ** 0.5

        IVT_objects = ar_ivt_tracking(IVT,
                                    IVTtrheshold,
                                    MinTimeIVT,
                                    dT,
                                    connectLon)

        grIVTs = calc_object_characteristics(IVT_objects, # feature object file
                                     IVT,         # original file used for feature detection
                                     OutputFolder+'IVT_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                                     Time,            # timesteps of the data
                                     Lat,             # 2D latidudes
                                     Lon,             # 2D Longitudes
                                     Gridspacing,
                                     Area,
                                     min_tsteps=int(MinTimeIVT/dT))      # minimum livetime in hours
        
        print('        check if MSs quallify as ARs')
        AR_obj = ar_check(IVT_objects,
                         AR_Lat,
                         AR_width_lenght_ratio,
                         AR_MinLen,
                         Lon,
                         Lat)
    
        grACs = calc_object_characteristics(AR_obj, # feature object file
                         IVT,         # original file used for feature detection
                         OutputFolder+'ARs_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                         Time,            # timesteps of the data
                         Lat,             # 2D latidudes
                         Lon,             # 2D Longitudes
                         Gridspacing,
                         Area)
        
       end = time.perf_counter()
       timer(start, end)
    
    if front_test == 'yes':
        print('======> identify frontal zones')
        start = time.perf_counter()
        
        # -------
        dx = dLon
        dy = dLat
        du = np.gradient( np.array(u850) )
        dv = np.gradient( np.array(v850) )
        PV = np.abs( dv[-1]/dx[None,:] - du[-2]/dy[None,:] )
        vgrad = np.gradient(np.array(t850), axis=(1,2))
        Tgrad = np.sqrt(vgrad[0]**2 + vgrad[1]**2)

        Fstar = PV * Tgrad

        Tgrad_zero = 0.45 #*100/(np.mean([dLon,dLat], axis=0)/1000.)  # 0.45 K/(100 km)
        import metpy.calc as calc
        from metpy.units import units
        CoriolisPar = np.array(calc.coriolis_parameter(np.deg2rad(Lat)))
        Frontal_Diagnostic = np.array(Fstar/(CoriolisPar * Tgrad_zero))
        
        FrontMask = np.copy(Mask)
        FrontMask[np.abs(Lat) < 10] = 0
        
        Frontal_Diagnostic = np.abs(Frontal_Diagnostic)
        Frontal_Diagnostic[:,FrontMask == 0] = 0
        # -------
        
        
        FR_objects = frontal_identification(Frontal_Diagnostic,
                              front_treshold,
                              MinAreaFR,
                              Area)
        
        end = time.perf_counter()
        timer(start, end)
        
        
    if slp_test == 'yes':
        print('======> track cyclones from PSL')
        start = time.perf_counter()
        
        CY_objects, ACY_objects= cy_acy_psl_tracking(
                                                    slp,
                                                    MaxPresAnCY,
                                                    MinTimeCY,
                                                    MinPresAnACY,
                                                    MinTimeACY,
                                                    dT,
                                                    Gridspacing,
                                                    connectLon
                                                    )

        grCyclonesPT = calc_object_characteristics(CY_objects, # feature object file
                                         slp,         # original file used for feature detection
                                         OutputFolder+'CY_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                                         Time,            # timesteps of the data
                                         Lat,             # 2D latidudes
                                         Lon,             # 2D Longitudes
                                         Gridspacing,
                                         Area,
                                         min_tsteps=int(MinTimeCY/dT)) 

        grACyclonesPT = calc_object_characteristics(ACY_objects, # feature object file
                                         slp,         # original file used for feature detection
                                         OutputFolder+'ACY_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                                         Time,            # timesteps of the data
                                         Lat,             # 2D latidudes
                                         Lon,             # 2D Longitudes
                                         Gridspacing,
                                         Area,
                                         min_tsteps=int(MinTimeCY/dT)) 

        end = time.perf_counter()
        timer(start, end)


    if z500_test == 'yes':
        print('======> track cyclones from Z500')
        start = time.perf_counter()
        
        cy_z500_objects, acy_z500_objects = cy_acy_z500_tracking(
                                            z500,
                                            MinTimeCY,
                                            dT,
                                            Gridspacing,
                                            connectLon
                                            )
        
        cy_z500_objects_characteristics = calc_object_characteristics(cy_z500_objects, # feature object file
                                     z500,         # original file used for feature detection
                                     OutputFolder+'CY-z500_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                                     Time,            # timesteps of the data
                                     Lat,             # 2D latidudes
                                     Lon,             # 2D Longitudes
                                     Gridspacing,
                                     Area,
                                     min_tsteps=int(MinTimeCY/dT))
        
        acy_z500_objects_characteristics = calc_object_characteristics(acy_z500_objects, # feature object file
                                 z500,         # original file used for feature detection
                                 OutputFolder+'ACY-z500_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                                 Time,            # timesteps of the data
                                 Lat,             # 2D latidudes
                                 Lon,             # 2D Longitudes
                                 Gridspacing,
                                 Area,
                                 min_tsteps=int(MinTimeCY/dT)) 
        
        print('    Check if cyclones qualify as Cut Off Low (COL)')
        col_obj = col_identification(cy_z500_objects,
                               z500,
                               u200,
                               Frontal_Diagnostic,
                               MinTimeC,
                               dx,
                               dy,
                               Lon,
                               Lat)
        
        col_stats = calc_object_characteristics(col_obj, # feature object file
                         z500*9.81,            # original file used for feature detection
                         OutputFolder+'COL_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                         Time,            # timesteps of the data
                         Lat,             # 2D latidudes
                         Lon,             # 2D Longitudes
                         Gridspacing,
                         Area,
                         min_tsteps=1)      # minimum livetime in hours

        end = time.perf_counter()
        timer(start, end)


    if mcs_tb_test == 'yes':
        print("======> 'check if Tb objects quallify as MCS (or selected storm type)")
        start = time.perf_counter()
        MCS_objects_Tb, C_objects = mcs_tb_tracking(tb,
                            pr,
                            SmoothSigmaC,
                            Pthreshold,
                            CL_Area,
                            CL_MaxT,
                            Cthreshold,
                            MinAreaC,
                            MinTimeC,
                            MCS_minPR,
                            MCS_minTime,
                            MCS_Minsize,
                            dT,
                            Area,
                            connectLon,
                           )
        
        grCs = calc_object_characteristics(C_objects, # feature object file
                             tb,         # original file used for feature detection
                             OutputFolder+'Clouds_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                             Time,            # timesteps of the data
                             Lat,             # 2D latidudes
                             Lon,             # 2D Longitudes
                             Gridspacing,
                             Area,
                             min_tsteps=int(MinTimeC/dT))      # minimum livetime in hours
        
        grMCSs_Tb = calc_object_characteristics(
            MCS_objects_Tb,  # feature object file
            pr,  # original file used for feature detection
            OutputFolder+'MCSs_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
            Time,            # timesteps of the data
            Lat,             # 2D latidudes
            Lon,             # 2D Longitudes
            Gridspacing,
            Area)
        
        end = time.perf_counter()
        timer(start, end)

#     if (mcs_tb_test == 'yes') & (ar_test == 'yes'):
#         start = time.perf_counter()
        
#         PR_objects, MCS_obj = mcs_pr_tracking(pr,
#                             tb,
#                             C_objects,
#                             AR_obj,
#                             Area,
#                             Lon,
#                             Lat,
#                             SmoothSigmaP,
#                             Pthreshold,
#                             MinTimePR,
#                             MCS_minPR,
#                             MCS_Minsize,
#                             CL_Area,
#                             CL_MaxT,
#                             MCS_minTime,
#                             MinAreaPR,
#                             dT,
#                             connectLon)
        
#         grPRs = calc_object_characteristics(PR_objects, # feature object file
#                                  pr,         # original file used for feature detection
#                                  OutputFolder+'PR_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
#                                  Time,            # timesteps of the data
#                                  Lat,             # 2D latidudes
#                                  Lon,             # 2D Longitudes
#                                  Gridspacing,
#                                  Area,
#                                  min_tsteps=int(MinTimePR/dT)) 
        
#         grMCSs = calc_object_characteristics(MCS_obj, # feature object file
#                              pr,         # original file used for feature detection
#                              OutputFolder+'MCSs_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
#                              Time,            # timesteps of the data
#                              Lat,             # 2D latidudes
#                              Lon,             # 2D Longitudes
#                              Gridspacing,
#                              Area)
        
#         end = time.perf_counter()
#         timer(start, end)
    
    
    if tc_test == 'yes':
        print('======> Check if cyclones qualify as TCs')
        start = time.perf_counter()
        
        TC_obj, TC_Tracks = tc_tracking(CY_objects,
                        t850,
                        slp,
                        tb,
                        C_objects,
                        Lon,
                        Lat,
                        TC_lat_genesis,
                        TC_deltaT_core,
                        TC_T850min,
                        TC_Pmin,
                        TC_lat_max
                       )
        
        grTCs = calc_object_characteristics(TC_obj, # feature object file
                             slp*100.,         # original file used for feature detection
                             OutputFolder+'TC_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                             Time,            # timesteps of the data
                             Lat,             # 2D latidudes
                             Lon,             # 2D Longitudes
                             Gridspacing,
                             Area,
                             min_tsteps=int(MinTimeC/dT))      # minimum livetime in hours
        ### SAVE THE TC TRACKS TO PICKL FILE
        a_file = open(OutputFolder+str(Time[0].year)+str(Time[0].month).zfill(2)+'_TCs_tracks.pkl', "wb")
        pickle.dump(TC_Tracks, a_file)
        a_file.close()
        
        end = time.perf_counter()
        timer(start, end)  
    
    


    print(' ')
    print('Save the object masks into a joint netCDF')
    start = time.perf_counter()
    # ============================
    # Write NetCDF
    iTime = np.array((Time - Time[0]).total_seconds()).astype('int')

    dataset = Dataset(NCfile,'w',format='NETCDF4_CLASSIC')
    yc = dataset.createDimension('yc', Lat.shape[0])
    xc = dataset.createDimension('xc', Lat.shape[1])
    time = dataset.createDimension('time', None)

    times = dataset.createVariable('time', np.float64, ('time',))
    lat = dataset.createVariable('lat', np.float32, ('yc','xc',))
    lon = dataset.createVariable('lon', np.float32, ('yc','xc',))
    if mcs_tb_test == 'yes':
        PR_real = dataset.createVariable('PR', np.float32,('time','yc','xc'),zlib=True)
        # PR_obj = dataset.createVariable('PR_Objects', np.float32,('time','yc','xc'),zlib=True)
        # MCSs = dataset.createVariable('MCS_Objects', np.float32,('time','yc','xc'),zlib=True)
        MCSs_Tb = dataset.createVariable('MCS_Tb_Objects', np.float32,('time','yc','xc'),zlib=True)
        Cloud_real = dataset.createVariable('BT', np.float32,('time','yc','xc'),zlib=True)
        Cloud_obj = dataset.createVariable('BT_Objects', np.float32,('time','yc','xc'),zlib=True)
    if front_test == 'yes':
        FR_real = dataset.createVariable('FR', np.float32,('time','yc','xc'),zlib=True)
        FR_obj = dataset.createVariable('FR_Objects', np.float32,('time','yc','xc'),zlib=True)
        T_real = dataset.createVariable('T850', np.float32,('time','yc','xc'),zlib=True)
    if slp_test == 'yes':
        CY_obj = dataset.createVariable('CY_Objects', np.float32,('time','yc','xc'),zlib=True)
        ACY_obj = dataset.createVariable('ACY_Objects', np.float32,('time','yc','xc'),zlib=True)
        SLP_real = dataset.createVariable('SLP', np.float32,('time','yc','xc'),zlib=True)
    if tc_test == 'yes':
        TCs = dataset.createVariable('TC_Objects', np.float32,('time','yc','xc'),zlib=True)
    if ms_test == 'yes':
        MS_real = dataset.createVariable('MS', np.float32,('time','yc','xc'),zlib=True)
        MS_obj = dataset.createVariable('MS_Objects', np.float32,('time','yc','xc'),zlib=True)
    if ar_test == 'yes':
        IVT_real = dataset.createVariable('IVT', np.float32,('time','yc','xc'),zlib=True)
        IVT_obj = dataset.createVariable('IVT_Objects', np.float32,('time','yc','xc'),zlib=True)
        ARs = dataset.createVariable('AR_Objects', np.float32,('time','yc','xc'),zlib=True)
    if z500_test == 'yes':
        CY_z500_obj = dataset.createVariable('CY_z500_Objects', np.float32,('time','yc','xc'),zlib=True)
        ACY_z500_obj = dataset.createVariable('ACY_z500_Objects', np.float32,('time','yc','xc'),zlib=True)
        Z500_real = dataset.createVariable('Z500', np.float32,('time','yc','xc'),zlib=True)
    if col_test == 'yes':
        COL = dataset.createVariable('COL_Objects', np.float32,('time','yc','xc'),zlib=True)
    if jet_test == 'yes':
        JET = dataset.createVariable('JET_Objects', np.float32,('time','yc','xc'),zlib=True)
        UV200 = dataset.createVariable('UV200', np.float32,('time','yc','xc'),zlib=True)
    if ew_test == 'yes':
        MRG = dataset.createVariable('MRG_Objects', np.float32,('time','yc','xc'),zlib=True)
        IGW = dataset.createVariable('IGW_Objects', np.float32,('time','yc','xc'),zlib=True)
        KELVIN = dataset.createVariable('Kelvin_Objects', np.float32,('time','yc','xc'),zlib=True)
        EIG = dataset.createVariable('EIG0_Objects', np.float32,('time','yc','xc'),zlib=True)
        ER = dataset.createVariable('ER_Objects', np.float32,('time','yc','xc'),zlib=True)
        

    times.calendar = "standard"
    times.units = "seconds since "+str(Time[0].year)+"-"+str(Time[0].month).zfill(2)+"-"+str(Time[0].day).zfill(2)+" "+str(Time[0].hour).zfill(2)+":"+str(Time[0].minute).zfill(2)+":00"
    times.standard_name = "time"
    times.long_name = "time"

    lat.long_name = "latitude" ;
    lat.units = "degrees_north" ;
    lat.standard_name = "latitude" ;

    lon.long_name = "longitude" ;
    lon.units = "degrees_east" ;
    lon.standard_name = "longitude" ;

    if mcs_tb_test == 'yes':
        PR_real.coordinates = "lon lat"
        PR_real.longname = "precipitation"
        PR_real.unit = "mm/"+str(dT)+"h"
        
#         PR_obj.coordinates = "lon lat"
#         PR_obj.longname = "precipitation objects"
#         PR_obj.unit = ""
        
#         MCSs.coordinates = "lon lat"
#         MCSs.longname = "MCSs object defined by their precipitation"
#         MCSs.unit = ""
        
        MCSs_Tb.coordinates = "lon lat"
        MCSs_Tb.longname = "MCSs object defined by their Tb"
        MCSs_Tb.unit = ""
        
        Cloud_real.coordinates = "lon lat"
        Cloud_real.longname = "Tb"
        Cloud_real.unit = "K"
        
        Cloud_obj.coordinates = "lon lat"
        Cloud_obj.longname = "Tb objects"
        Cloud_obj.unit = ""
    if front_test == 'yes':
        FR_real.coordinates = "lon lat"
        FR_real.longname = "frontal index"
        FR_real.unit = ""
        
        FR_obj.coordinates = "lon lat"
        FR_obj.longname = "frontal objects"
        FR_obj.unit = ""
        
        T_real.coordinates = "lon lat"
        T_real.longname = "850 hPa air temperature"
        T_real.unit = "K"
    if slp_test == 'yes':
        CY_obj.coordinates = "lon lat"
        CY_obj.longname = "cyclone objects from SLP"
        CY_obj.unit = ""
        
        ACY_obj.coordinates = "lon lat"
        ACY_obj.longname = "anticyclone objects from SLP"
        ACY_obj.unit = ""
        
        SLP_real.coordinates = "lon lat"
        SLP_real.longname = "sea level pressure (SLP)"
        SLP_real.unit = "Pa"
    if ms_test == 'yes':
        MS_real.coordinates = "lon lat"
        MS_real.longname = "850 hPa moisture flux"
        MS_real.unit = "g/g m/s"
        
        MS_obj.coordinates = "lon lat"
        MS_obj.longname = "mosture streams objects according to 850 hPa moisture flux"
        MS_obj.unit = ""
    if ar_test == 'yes':
        IVT_real.coordinates = "lon lat"
        IVT_real.longname = "vertically integrated moisture transport"
        IVT_real.unit = "kg m−1 s−1"
        
        IVT_obj.coordinates = "lon lat"
        IVT_obj.longname = "IVT objects"
        IVT_obj.unit = ""
        
        ARs.coordinates = "lon lat"
        ARs.longname = "atmospheric river objects"
        ARs.unit = ""
    if tc_test == 'yes':
        TCs.coordinates = "lon lat"
        TCs.longname = "tropical cyclone objects"
        TCs.unit = ""
    if z500_test == 'yes':
        CY_z500_obj.coordinates = "lon lat"
        CY_z500_obj.longname = "cyclone objects according to Z500"
        CY_z500_obj.unit = ""
        
        ACY_z500_obj.coordinates = "lon lat"
        ACY_z500_obj.longname = "anticyclone objects according to Z500"
        ACY_z500_obj.unit = ""
        
        Z500_real.coordinates = "lon lat"
        Z500_real.longname = "500 hPa geopotential height"
        Z500_real.unit = "gpm"
    if col_test == 'yes':
        COL.coordinates = "lon lat"
        COL.longname = "cut off low objects"
        COL.unit = ""
    if jet_test == 'yes':
        JET.coordinates = "lon lat"
        JET.longname = "jet stream objects"
        JET.unit = ""
        
        UV200.coordinates = "lon lat"
        UV200.longname = "200 hPa wind speed"
        UV200.unit = "m s-1"
    if ew_test == 'yes':
        MRG.coordinates = "lon lat"
        MRG.longname = "Mixed Rosby Gravity wave objects"
        MRG.unit = ""
        
        IGW.coordinates = "lon lat"
        IGW.longname = "Inertia Gravity wave objects"
        IGW.unit = ""
        
        KELVIN.coordinates = "lon lat"
        KELVIN.longname = "Kelvin wave objects"
        KELVIN.unit = ""
        
        EIG.coordinates = "lon lat"
        EIG.longname = "Eastward Inertio Gravirt wave objects"
        EIG.unit = ""
        
        ER.coordinates = "lon lat"
        ER.longname = "Equatorial Rossby wave objects"
        ER.unit = ""

    lat[:] = Lat
    lon[:] = Lon
    if mcs_tb_test == 'yes':
        PR_real[:] = pr
        # PR_obj[:] = PR_objects
        # MCSs[:] = MCS_obj
        MCSs_Tb[:] = MCS_objects_Tb
        Cloud_real[:] = tb
        Cloud_obj[:] = C_objects
    if front_test == 'yes':
        FR_real[:] = Frontal_Diagnostic
        FR_obj[:] = FR_objects
        T_real[:] = t850
    if tc_test == 'yes':
        TCs[:] = TC_obj
    if slp_test == 'yes':
        CY_obj[:] = CY_objects
        ACY_obj[:] = ACY_objects
        SLP_real[:] = slp
    if ms_test == 'yes':
        MS_real[:] = VapTrans
        MS_obj[:] = MS_objects
    if ar_test == 'yes':
        IVT_real[:] = IVT
        IVT_obj[:] = IVT_objects
        ARs[:] = AR_obj
    if z500_test == 'yes':
        CY_z500_obj[:] = cy_z500_objects
        ACY_z500_obj[:] = acy_z500_objects
        Z500_real[:] = z500
    if col_test == 'yes':
        COL[:] = col_obj
    if jet_test == 'yes':
        JET[:] = jet_objects
        UV200[:] = uv200
    if ew_test == 'yes':
        MRG[:] = mrg_objects
        IGW[:] = igw_objects
        KELVIN[:] = kelvin_objects
        EIG[:] = eig0_objects
        ER[:] = er_objects
                
    times[:] = iTime

    dataset.close()
    print('Saved: '+NCfile)
    import time
    end = time.perf_counter()
    timer(start, end)

    if tc_test == 'yes':
        ### SAVE THE TC TRACKS TO PICKL FILE
        # ============================
        a_file = open(OutputFolder+str(Time[0].year)+str(Time[0].month).zfill(2)+'_TCs_tracks.pkl', "wb")
        pickle.dump(TC_Tracks, a_file)
        a_file.close()
    try:
        object_split
    except NameError:
        object_split = False    
    return object_split
