# import os
# import sys
# sys.path.append('../')
# import numpy as np
# import xarray as xr
# import pymk as pymk
# import time

# start_time1 = time.time()

# def pmk_test(ts):
#     if np.isnan(ts).all():
#         return np.nan, np.nan, np.nan, np.nan
#     results = pymk.yue_wang_modification_test(ts)
#     slope, p_value, n_ns, intercept = results.slope, results.p, results.n_ns, results.intercept
#     return slope, p_value, n_ns, intercept

# uovo_alb = xr.open_dataset('../../data/006_004_medmfc_uovo_alb.nc')
# uo_surf = uovo_alb.uo[:,0,:,:].sel(time=slice(None, '2023-06-07'))
# vo_surf = uovo_alb.vo[:,0,:,:].sel(time=slice(None, '2023-06-07'))
# uoa_surf = uo_surf - uo_surf.mean(dim='time')
# voa_surf = vo_surf - vo_surf.mean(dim='time')
# eke_surf_alb = 0.5 * (uoa_surf**2 + voa_surf**2)*10000

# a = [np.zeros((len(eke_surf_alb.latitude), len(eke_surf_alb.longitude))) for _ in range(4)]
# sa, pa, na, ia = a


# for i in range(len(eke_surf_alb.latitude)):
#     for j in range(len(eke_surf_alb.longitude)):
#         sa[i,j], pa[i,j], na[i,j], ia[i,j] = pmk_test(eke_surf_alb[:,i,j])

# AG = {'slope': sa, 'pvalue': pa, 'intercept': ia, 'nns': na}

# np.save('../../results/model_alb_slpintns.npy', AG)

# end_time1 = time.time()
# execution_time1 = end_time1 - start_time1
# print(f"Time: {execution_time1} seconds")


# uovo_iera = xr.open_dataset('../../data/006_004_medmfc_uovo_iera.nc')
# uo_surf = uovo_iera.uo.sel(time=slice(None, '2023-06-07'))
# vo_surf = uovo_iera.vo.sel(time=slice(None, '2023-06-07'))
# uoa_surf = uo_surf - uo_surf.mean(dim='time')
# voa_surf = vo_surf - vo_surf.mean(dim='time')
# eke_surf_iera = 0.5 * (uoa_surf**2 + voa_surf**2)*10000

# a = [np.zeros((len(eke_surf_iera.latitude), len(eke_surf_iera.longitude))) for _ in range(4)]
# sa, pa, na, ia = a

# for i in range(len(eke_surf_iera.latitude)):
#     for j in range(len(eke_surf_iera.longitude)):
#         sa[i,j], pa[i,j], na[i,j], ia[i,j] = pmk_test(eke_surf_iera[:,i,j])

# AG = {'slope': sa, 'pvalue': pa, 'intercept': ia, 'nns': na}

# np.save('../../results/model_iera_slpintns.npy', AG)

# end_time2 = time.time()
# execution_time2 = end_time2 - start_time1
# print(f"Time: {execution_time2} seconds")




##################################################################################################
##################################################################################################
#Versión más rápida

# import sys
# sys.path.append('../')

# import numpy as np
# import xarray as xr
# import pymk
# import time
# from joblib import Parallel, delayed


# # ============================================================
# # PARAMETRES
# # ============================================================

# END_DATE = '2023-06-07'

# N_JOBS = 6   # -1 = utilise tous les CPU disponibles


# # ============================================================
# # TEST MK POUR UN PIXEL
# # ============================================================

# def pmk_test(ts):

#     # Pas assez de données
#     if np.isnan(ts).all():
#         return np.nan, np.nan, np.nan, np.nan

#     results = pymk.yue_wang_modification_test(ts)

#     return (
#         results.slope,
#         results.p,
#         results.n_ns,
#         results.intercept
#     )


# # ============================================================
# # CALCUL MK SUR UNE GRILLE
# # ============================================================

# def compute_mk_eke(
#         filepath,
#         output,
#         surface=False,
#         depth_index=0):

#     print('\n' + '=' * 70)
#     print(f'Processing: {filepath}')
#     print('=' * 70)

#     start = time.time()

#     # --------------------------------------------------------
#     # OUVERTURE
#     # --------------------------------------------------------

#     ds = xr.open_dataset(filepath)

#     # --------------------------------------------------------
#     # SELECTION TEMPORELLE
#     # --------------------------------------------------------

#     ds = ds.sel(time=slice(None, END_DATE))

#     print(
#         f"Time range: "
#         f"{ds.time.values[0]} -> {ds.time.values[-1]}"
#     )

#     # --------------------------------------------------------
#     # SELECTION DE LA SURFACE
#     # --------------------------------------------------------

#     if surface:
#         uo = ds.uo.isel(depth=depth_index)
#         vo = ds.vo.isel(depth=depth_index)
#     else:
#         uo = ds.uo
#         vo = ds.vo

#     # --------------------------------------------------------
#     # CHARGEMENT EN MEMOIRE
#     # --------------------------------------------------------

#     print("Loading velocity data...")

#     uo = uo.load().values
#     vo = vo.load().values

#     print("uo shape:", uo.shape)
#     print("vo shape:", vo.shape)

#     ds.close()

#     # --------------------------------------------------------
#     # ANOMALIES
#     # --------------------------------------------------------

#     print("Computing velocity anomalies...")

#     uo = uo - np.nanmean(uo, axis=0)
#     vo = vo - np.nanmean(vo, axis=0)

#     # --------------------------------------------------------
#     # EKE
#     # --------------------------------------------------------

#     print("Computing EKE...")

#     eke = 0.5 * (uo**2 + vo**2) * 10000

#     nt, nlat, nlon = eke.shape

#     print(f"EKE shape: {eke.shape}")
#     print(f"Number of spatial points: {nlat * nlon:,}")

#     # --------------------------------------------------------
#     # PREPARATION DES SERIES
#     # --------------------------------------------------------

#     # Chaque élément correspond à un pixel (i,j)
#     time_series = [
#         eke[:, i, j]
#         for i in range(nlat)
#         for j in range(nlon)
#     ]

#     # --------------------------------------------------------
#     # CALCUL PARALLELE DU MK
#     # --------------------------------------------------------

#     print(
#         f"Running Yue-Wang MK test "
#         f"using {N_JOBS if N_JOBS > 0 else 'all available'} CPUs..."
#     )

#     results = Parallel(
#         n_jobs=N_JOBS,
#         backend='loky',
#         verbose=10
#     )(
#         delayed(pmk_test)(ts)
#         for ts in time_series
#     )

#     # --------------------------------------------------------
#     # CONVERSION EN ARRAY
#     # --------------------------------------------------------

#     results = np.asarray(results)

#     slope = results[:, 0].reshape(nlat, nlon)
#     pvalue = results[:, 1].reshape(nlat, nlon)
#     nns = results[:, 2].reshape(nlat, nlon)
#     intercept = results[:, 3].reshape(nlat, nlon)

#     # --------------------------------------------------------
#     # SAUVEGARDE
#     # --------------------------------------------------------

#     AG = {
#         'slope': slope,
#         'pvalue': pvalue,
#         'intercept': intercept,
#         'nns': nns
#     }

#     np.save(output, AG)

#     print(f"\nSaved: {output}")
#     print(f"Total time: {time.time() - start:.1f} s")

#     return AG


# # ============================================================
# # ALBORAN
# # ============================================================

# start_total = time.time()

# AG_alb = compute_mk_eke(
#     filepath='../../data/006_004_medmfc_uovo_alb.nc',
#     output='../../results/model_alb_slpintns.npy',
#     surface=True,
#     depth_index=0
# )


# # ============================================================
# # IERAPETRA
# # ============================================================

# # AG_iera = compute_mk_eke(
# #     filepath='../../data/006_004_medmfc_uovo_iera.nc',
# #     output='../../results/model_iera_slpintns.npy',
# #     surface=False
# # )
# # No funcionó por el espacio


# # ============================================================
# # TEMPS TOTAL
# # ============================================================

# print('\n' + '=' * 70)
# print(
#     f'Total execution time: '
#     f'{time.time() - start_total:.1f} seconds'
# )
# print('=' * 70)







#######################################################################################################
#######################################################################################################

import sys
sys.path.append('../')
import os
import time
import numpy as np
import xarray as xr
import pymk
from joblib import Parallel, delayed




# ============================================================
# PARAMETERS
# ============================================================

N_JOBS = 4              # 4 est un bon compromis pour ta machine
BLOCK_LAT = 20          # nombre de latitudes traitées simultanément
BLOCK_LON = 20          # nombre de longitudes traitées simultanément

DATE_END = "2023-06-07"


# ============================================================
# MANN-KENDALL
# ============================================================

def pmk_test(ts):

    # Original behavior
    if np.isnan(ts).all():
        return np.nan, np.nan, np.nan, np.nan

    results = pymk.yue_wang_modification_test(ts)

    slope = results.slope
    p_value = results.p
    n_ns = results.n_ns
    intercept = results.intercept

    return slope, p_value, n_ns, intercept


# ============================================================
# PROCESS ONE SPATIAL BLOCK
# ============================================================

def process_block(uo_block, vo_block):

    """
    uo_block, vo_block:
        dimensions = (time, lat, lon)

    Returns:
        slope, pvalue, nns, intercept
    """

    # --------------------------------------------------------
    # Temporal mean
    # --------------------------------------------------------

    u_mean = np.nanmean(uo_block, axis=0)
    v_mean = np.nanmean(vo_block, axis=0)

    # --------------------------------------------------------
    # Anomalies
    # --------------------------------------------------------

    u_anom = uo_block - u_mean[None, :, :]
    v_anom = vo_block - v_mean[None, :, :]

    # --------------------------------------------------------
    # EKE
    #
    # factor 10000 converts:
    # m2/s2 -> cm2/s2
    # --------------------------------------------------------

    eke = 0.5 * (u_anom**2 + v_anom**2) * 10000.

    nlat = eke.shape[1]
    nlon = eke.shape[2]

    # --------------------------------------------------------
    # Create list of pixel time series
    # --------------------------------------------------------

    time_series = [
        eke[:, i, j]
        for i in range(nlat)
        for j in range(nlon)
    ]

    # --------------------------------------------------------
    # MK tests
    #
    # IMPORTANT:
    # only small 1-D arrays are sent to workers.
    # The entire NetCDF / EKE field is NOT copied.
    # --------------------------------------------------------

    results = Parallel(
        n_jobs=N_JOBS,
        backend="loky",
        verbose=0
    )(
        delayed(pmk_test)(ts)
        for ts in time_series
    )

    # --------------------------------------------------------
    # Reshape results
    # --------------------------------------------------------

    results = np.asarray(results, dtype=float)

    slope = results[:, 0].reshape(nlat, nlon)
    pvalue = results[:, 1].reshape(nlat, nlon)
    nns = results[:, 2].reshape(nlat, nlon)
    intercept = results[:, 3].reshape(nlat, nlon)

    return slope, pvalue, nns, intercept


# ============================================================
# MAIN FUNCTION
# ============================================================

def compute_mk_eke(
    filepath,
    output,
    surface=False,
    depth_index=0
):

    start_time = time.time()

    print("\n" + "=" * 70)
    print("FILE:", filepath)
    print("=" * 70)

    # --------------------------------------------------------
    # Open lazily
    #
    # NOTHING is loaded into RAM here.
    # --------------------------------------------------------

    ds = xr.open_dataset(filepath)

    print(ds)

    # --------------------------------------------------------
    # Select time
    # --------------------------------------------------------

    ds = ds.sel(time=slice(None, DATE_END))

    # --------------------------------------------------------
    # Select velocity
    # --------------------------------------------------------


    print("Selecting surface level:", depth_index)

    uo = ds.ug
    vo = ds.vg
    # uo = ds.uo.isel(depth=depth_index)
    # vo = ds.vo.isel(depth=depth_index)

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    print("\nuo dimensions:", uo.dims)
    print("uo shape:", uo.shape)

    if uo.ndim != 3:
        raise ValueError(
            f"uo must have dimensions (time, latitude, longitude). "
            f"Got {uo.dims} with shape {uo.shape}"
        )

    lat_dim = "latitude"
    lon_dim = "longitude"
    time_dim = "time"

    nlat = uo.sizes[lat_dim]
    nlon = uo.sizes[lon_dim]

    print(f"\nGrid: {nlat} x {nlon}")
    print(f"Block: {BLOCK_LAT} x {BLOCK_LON}")
    print(f"Parallel jobs: {N_JOBS}")

    # --------------------------------------------------------
    # Output arrays
    #
    # These are tiny compared with the NetCDF.
    # --------------------------------------------------------

    slope = np.full(
        (nlat, nlon),
        np.nan,
        dtype=np.float64
    )

    pvalue = np.full(
        (nlat, nlon),
        np.nan,
        dtype=np.float64
    )

    nns = np.full(
        (nlat, nlon),
        np.nan,
        dtype=np.float64
    )

    intercept = np.full(
        (nlat, nlon),
        np.nan,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Loop over spatial blocks
    # --------------------------------------------------------

    total_blocks = (
        int(np.ceil(nlat / BLOCK_LAT))
        *
        int(np.ceil(nlon / BLOCK_LON))
    )

    block_number = 0

    for i0 in range(0, nlat, BLOCK_LAT):

        i1 = min(i0 + BLOCK_LAT, nlat)

        for j0 in range(0, nlon, BLOCK_LON):

            j1 = min(j0 + BLOCK_LON, nlon)

            block_number += 1

            print(
                f"\nBlock {block_number}/{total_blocks} "
                f"lat={i0}:{i1}, lon={j0}:{j1}"
            )

            # ------------------------------------------------
            # READ ONLY THIS BLOCK
            #
            # .values triggers loading of only this subset.
            # ------------------------------------------------

            uo_block = (
                uo
                .isel(
                    {
                        lat_dim: slice(i0, i1),
                        lon_dim: slice(j0, j1)
                    }
                )
                .values
            )

            vo_block = (
                vo
                .isel(
                    {
                        lat_dim: slice(i0, i1),
                        lon_dim: slice(j0, j1)
                    }
                )
                .values
            )

            print(
                "Loaded block:",
                uo_block.shape,
                f"RAM arrays ≈ "
                f"{(uo_block.nbytes + vo_block.nbytes) / 1024**2:.1f} MB"
            )

            # ------------------------------------------------
            # MK calculation
            # ------------------------------------------------

            (
                block_slope,
                block_pvalue,
                block_nns,
                block_intercept
            ) = process_block(
                uo_block,
                vo_block
            )

            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            slope[i0:i1, j0:j1] = block_slope
            pvalue[i0:i1, j0:j1] = block_pvalue
            nns[i0:i1, j0:j1] = block_nns
            intercept[i0:i1, j0:j1] = block_intercept

            # ------------------------------------------------
            # Explicitly delete large arrays
            # ------------------------------------------------

            del uo_block
            del vo_block
            del block_slope
            del block_pvalue
            del block_nns
            del block_intercept

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    AG = {
        "slope": slope,
        "pvalue": pvalue,
        "intercept": intercept,
        "nns": nns
    }

    np.save(output, AG)

    ds.close()

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("DONE")
    print("Output:", output)
    print(f"Time: {elapsed / 60:.1f} min")
    print("=" * 70)


# ============================================================
# ALBORAN
# ============================================================

compute_mk_eke(
    filepath="../../results/006_004_medmfc_vg_alb.nc",
    output="../../results/model_albvg_slpintns.npy",
    surface=True,
    depth_index=0
)

# ============================================================
# IERAPETRA
# ============================================================

compute_mk_eke(
    filepath="../../results/006_004_medmfc_vg_iera.nc",
    output="../../results/model_ieravg_slpintns.npy",
    surface=True,
    depth_index=0
)