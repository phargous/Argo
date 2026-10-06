import numpy as np
import xarray as xr
from scipy import ndimage
import os


def dirtodict(dirPath):
    d = {}
    for i in [os.path.join(dirPath, i) for i in os.listdir(dirPath)
              if os.path.isdir(os.path.join(dirPath, i))]:
        d[os.path.basename(i)] = dirtodict(i)
    d['.files'] = [os.path.join(dirPath, i) for i in os.listdir(dirPath)
                   if os.path.isfile(os.path.join(dirPath, i))]
    return d



region_dic = {
              # 'CA': [[25, -23], [30, -10]],
              'WMED': [[35, -5.5], [45, 14.9]],
              'EMED': [[30, 15], [40, 36]],
              # 'BS': [[41, 28], [46, 41.5]],
              # 'GS': [[40, -28], [60, -10]],
              # 'CHIS': [[48, -5], [59, 5]],
             }


def smooth_regions(da, lim = 1000):
  
    np_da = da.values

    # Create a structure for the morphological operations
    struct = ndimage.generate_binary_structure(2, 2)

    # Fill the holes
    np_da_filled = ndimage.binary_fill_holes(np_da, structure=struct).astype(int)

    # Label each separate region
    label_np_da, num_features = ndimage.label(np_da_filled)

    # Create a size map
    sizes = ndimage.sum(np_da_filled, label_np_da, range(num_features + 1))

    # Remove small regions that have sizes less than a threshold 
    mask_sizes = sizes > lim
    np_da_clean = mask_sizes[label_np_da]

    smooth_np_da = ndimage.binary_closing(np_da_clean, structure=struct).astype(int)

    # Convert the numpy array back to DataArray
    da_filled_clean = xr.DataArray(smooth_np_da, coords=da.coords, dims=da.dims)


    return da_filled_clean

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the haversine distance between two sets of coordinates.
    
    More info: https://en.wikipedia.org/wiki/Haversine_formula
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    r = 6371 # Earth radius in km 
    
    distance = r * c

    return distance

def get_area_haversine(lat, lon):
    """
    Compute area of a rectilinear grid on a sphere using the haversine formula.
    """
    
    lat2d, lon2d = np.meshgrid(lat, lon, indexing='ij')

    lat1 = lat2d[:-1, :-1]    
    lat2 = lat2d[1:, :-1]
    
    lon1 = lon2d[:-1, :-1] 
    lon2 = lon2d[:-1, 1:]
 
    dlat = haversine(lon1, lat1, lon1, lat2)
    dlon = haversine(lon1, lat1, lon2, lat1)

    
    # Add NaN at the end of each array axis to have the same 
    # shape as lat and lon.
    new_value = np.nan
    
    # Add a NaN value at the end of the latitudinal axis
    dy = np.append(dlat, np.full((1, dlat.shape[1]), new_value), axis=0)
    dx = np.append(dlon, np.full((1, dlon.shape[1]), new_value), axis=0)
    
    # Add a NaN value at the end of the longitudinal axis
    dyy = np.append(dy, np.full((dy.shape[0], 1), new_value), axis=1)
    dxx = np.append(dx, np.full((dx.shape[0], 1), new_value), axis=1)

    return dxx * dyy

def get_mean_haversine(da, mask):
    
    area            = get_area_haversine(mask['lat'], mask['lon'])
    da_area         = xr.Dataset(coords={'lat': mask.lat, 'lon': mask.lon})
    da_area['area'] = (('lat', 'lon'), area)
    area            = da_area['area'].where(mask==1)
    area_sum        = area.sum(['lat', 'lon']).compute()
    area_rat        = area/area_sum
    ds_r            = da.where(mask==1)
    ds_r            = (ds_r * area_rat).sum(['lat', 'lon']).compute()
    val             = ds_r.values

    return val

def get_significant_trend_area_ratio_haversine(eke_trend, masks, region):

    sign_trend_reg      = eke_trend.sign.where(masks[region].notnull())
    sign_trend_reg_pixs = sign_trend_reg.where(sign_trend_reg==1)

    area            = get_area_haversine(masks['lat'], masks['lon'])
    da_area         = xr.Dataset(coords={'lat': masks.lat, 'lon': masks.lon})
    da_area['area'] = (('lat', 'lon'), area)
    area            = da_area['area'].where(masks[region]==1)
    area_sum        = area.sum(['lat', 'lon']).compute()

    sign_trend_reg_area = (sign_trend_reg_pixs * area).sum(['lat', 'lon']).compute()

    ratio = np.float32(sign_trend_reg_area/area_sum)

    return ratio


def crop(ds, coords, drop=False, 
         coord_names = ['lat', 'lon']):

    ds_crop = ds.where((ds[coord_names[0]] > coords[0][0]) &
                       (ds[coord_names[0]] < coords[1][0]) &
                       (ds[coord_names[1]] > coords[0][1]) &
                       (ds[coord_names[1]] < coords[1][1]), drop=drop)

    return ds_crop




