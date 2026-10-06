import sys 
sys.path.append("../")
import os
import xarray as xr
import numpy as np


bathy_iera = xr.open_dataset('../../data/006_004_medmfc_bathy_ieratemp.nc')
uovo_iera = xr.open_dataset('../../data/006_004_medmfc_uovo_iera.nc')

valid_depth = bathy_iera.mask.any(dim=["latitude", "longitude"])
uovo_iera = uovo_iera.sel(depth=valid_depth)
uovo_iera.to_netcdf('../../data/006_004_medmfc_uovo_iera2.nc')