import sys
sys.path.append('../')

import os
import time
import numpy as np
import xarray as xr
import gsw
from netCDF4 import Dataset

from joblib import Parallel, delayed


# ============================================================
# PARAMETERS
# ============================================================
TEST_MODE = False
TEST_NTIME = 10000
TEST_NLAT = 20
TEST_NLON = 20

N_JOBS = 4

# Small spatial blocks
BLOCK_LAT = 20
BLOCK_LON = 20

# Depths wanted in final product
TARGET_DEPTHS = [50, 100, 200, 500, 1000]

# Time period
DATE_START = "1993-01-01"
DATE_END   = "2026-06-07"

# Climatology
CLIM_START = "1993-01-01"
CLIM_END   = "2012-12-31"

# Earth
OMEGA = 7.292115e-5
R_EARTH = 6371000.0


# ============================================================
# COMPUTE ONE T/S PROFILE
# ============================================================

def compute_dynamic_height_profile(
    SP_profile,
    theta_profile,
    lat,
    lon,
    depth
):
    """
    Compute dynamic height for ONE horizontal point.

    Input
    -----

    SP_profile :
        shape = (time, depth)

    theta_profile :
        shape = (time, depth)

    lat :
        scalar

    lon :
        scalar

    depth :
        shape = (depth,)

    Returns
    -------

    D :
        shape = (time, depth)

    """

    # --------------------------------------------------------
    # Pressure
    # --------------------------------------------------------

    p = gsw.p_from_z(
        -depth,
        lat
    )

    # --------------------------------------------------------
    # Broadcast pressure
    # --------------------------------------------------------

    p_2d = np.broadcast_to(
        p[None, :],
        SP_profile.shape
    )

    # --------------------------------------------------------
    # Longitude / latitude
    # --------------------------------------------------------

    lon_2d = np.full(
        SP_profile.shape,
        lon
    )

    lat_2d = np.full(
        SP_profile.shape,
        lat
    )

    # --------------------------------------------------------
    # Absolute Salinity
    # --------------------------------------------------------

    SA = gsw.SA_from_SP(
        SP_profile,
        p_2d,
        lon_2d,
        lat_2d
    )

    # --------------------------------------------------------
    # Conservative Temperature
    #
    # thetao = potential temperature
    # --------------------------------------------------------

    CT = gsw.CT_from_pt(
        SA,
        theta_profile
    )

    # --------------------------------------------------------
    # Dynamic height relative to surface
    # --------------------------------------------------------

    try:

        D = gsw.geo_strf_dyn_height(
            SA,
            CT,
            p,
            p_ref=0.0,
            axis=-1
        )

    except Exception:

        D = np.full(
            SP_profile.shape,
            np.nan,
            dtype=np.float64
        )

    return D


# ============================================================
# PROCESS ALL PIXELS OF ONE BLOCK
# ============================================================

def compute_dynamic_height_block(
    SP_block,
    theta_block,
    lat,
    lon,
    depth
):
    """
    Compute dynamic height for all pixels in one block.

    Input:
        SP_block
            (time, depth, lat, lon)

        theta_block
            (time, depth, lat, lon)

    Output:
        D_block
            (time, depth, lat, lon)
    """

    nt, nz, nlat, nlon = SP_block.shape

    # --------------------------------------------------------
    # Create pixel list
    # --------------------------------------------------------

    pixels = [
        (i, j)
        for i in range(nlat)
        for j in range(nlon)
    ]

    # --------------------------------------------------------
    # Parallel T/S calculations
    #
    # IMPORTANT:
    # Each worker receives only ONE vertical profile.
    # --------------------------------------------------------

    results = Parallel(
        n_jobs=N_JOBS,
        backend="loky",
        verbose=0
    )(
        delayed(compute_dynamic_height_profile)(
            SP_block[:, :, i, j],
            theta_block[:, :, i, j],
            lat[i],
            lon[j],
            depth
        )
        for i, j in pixels
    )

    # --------------------------------------------------------
    # Reconstruct block
    # --------------------------------------------------------

    D_block = np.full(
        (
            nt,
            nz,
            nlat,
            nlon
        ),
        np.nan,
        dtype=np.float64
    )

    for result, (i, j) in zip(
        results,
        pixels
    ):

        D_block[:, :, i, j] = result

    return D_block


# ============================================================
# HORIZONTAL GRADIENTS
# ============================================================

def compute_horizontal_gradients(
    D,
    lat,
    lon
):
    """
    Compute:

        dD/dy
        dD/dx

    using centered finite differences.

    D dimensions:

        time, depth, lat, lon
    """

    nt, nz, nlat, nlon = D.shape

    dDdy = np.full_like(
        D,
        np.nan
    )

    dDdx = np.full_like(
        D,
        np.nan
    )

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    # ========================================================
    # MERIDIONAL DERIVATIVE
    # ========================================================

    for i in range(nlat):

        if i == 0:

            dy = (
                lat_rad[i + 1]
                - lat_rad[i]
            ) * R_EARTH

            dDdy[:, :, i, :] = (
                D[:, :, i + 1, :]
                -
                D[:, :, i, :]
            ) / dy

        elif i == nlat - 1:

            dy = (
                lat_rad[i]
                -
                lat_rad[i - 1]
            ) * R_EARTH

            dDdy[:, :, i, :] = (
                D[:, :, i, :]
                -
                D[:, :, i - 1, :]
            ) / dy

        else:

            dy = (
                lat_rad[i + 1]
                -
                lat_rad[i - 1]
            ) * R_EARTH

            dDdy[:, :, i, :] = (
                D[:, :, i + 1, :]
                -
                D[:, :, i - 1, :]
            ) / dy

    # ========================================================
    # ZONAL DERIVATIVE
    # ========================================================

    for j in range(nlon):

        if j == 0:

            dx = (
                lon_rad[j + 1]
                -
                lon_rad[j]
            ) * R_EARTH

            dx = (
                dx
                * np.cos(lat_rad)
            )

            dDdx[:, :, :, j] = (
                D[:, :, :, j + 1]
                -
                D[:, :, :, j]
            ) / dx[None, None, :]

        elif j == nlon - 1:

            dx = (
                lon_rad[j]
                -
                lon_rad[j - 1]
            ) * R_EARTH

            dx = (
                dx
                * np.cos(lat_rad)
            )

            dDdx[:, :, :, j] = (
                D[:, :, :, j]
                -
                D[:, :, :, j - 1]
            ) / dx[None, None, :]

        else:

            dx = (
                lon_rad[j + 1]
                -
                lon_rad[j - 1]
            ) * R_EARTH

            dx = (
                dx
                * np.cos(lat_rad)
            )

            dDdx[:, :, :, j] = (
                D[:, :, :, j + 1]
                -
                D[:, :, :, j - 1]
            ) / dx[None, None, :]

    return dDdy, dDdx


# ============================================================
# GEOSTROPHIC VELOCITY
# ============================================================

def compute_geostrophic_velocity(
    D,
    lat
):
    """
    From dynamic height:

        ug = -(1/f) dD/dy
        vg = +(1/f) dD/dx

    Then remove the surface relative velocity and add
    the SSH-derived surface velocity later.
    """

    dDdy, dDdx = compute_horizontal_gradients(
        D,
        lat,
        np.arange(D.shape[-1])
    )

    # --------------------------------------------------------
    # Coriolis parameter
    # --------------------------------------------------------

    f = (
        2.0
        * OMEGA
        * np.sin(
            np.deg2rad(lat)
        )
    )

    f4d = f[
        None,
        None,
        :,
        None
    ]

    # --------------------------------------------------------
    # Relative geostrophic velocity
    # --------------------------------------------------------

    ug_rel = -dDdy / f4d
    vg_rel =  dDdx / f4d

    # --------------------------------------------------------
    # Remove surface velocity
    #
    # Dynamic height is referenced to surface, but because
    # of numerical horizontal derivatives we explicitly
    # enforce:
    #
    # ug_rel(surface) = 0
    # vg_rel(surface) = 0
    # --------------------------------------------------------

    ug0 = ug_rel[:, 0, :, :]
    vg0 = vg_rel[:, 0, :, :]

    ug_rel = (
        ug_rel
        - ug0[:, None, :, :]
    )

    vg_rel = (
        vg_rel
        - vg0[:, None, :, :]
    )

    return ug_rel, vg_rel


# ============================================================
# FIND DEPTH INDICES
# ============================================================

def find_depth_indices(
    depth,
    target_depths
):

    indices = []

    actual_depths = []

    for target in target_depths:

        idx = np.argmin(
            np.abs(
                depth - target
            )
        )

        indices.append(idx)
        actual_depths.append(
            depth[idx]
        )

    return (
        np.asarray(indices),
        np.asarray(actual_depths)
    )


# ============================================================
# PROCESS ONE SPATIAL BLOCK
# ============================================================

def process_block(
    so_ds,
    theta_ds,
    ssh_ds,
    i0,
    i1,
    j0,
    j1,
    depth,
    depth_indices,
    clim_mask
):

    # ========================================================
    # ADD ONE-CELL HALO
    #
    # Needed for horizontal derivatives.
    # ========================================================

    hi0 = max(
        0,
        i0 - 1
    )

    hi1 = min(
        so_ds.sizes["latitude"],
        i1 + 1
    )

    hj0 = max(
        0,
        j0 - 1
    )

    hj1 = min(
        so_ds.sizes["longitude"],
        j1 + 1
    )

    # ========================================================
    # COORDINATES
    # ========================================================

    lat = (
        so_ds.latitude
        .isel(
            latitude=slice(
                hi0,
                hi1
            )
        )
        .values
    )

    lon = (
        so_ds.longitude
        .isel(
            longitude=slice(
                hj0,
                hj1
            )
        )
        .values
    )

    # ========================================================
    # LOAD ONLY THIS BLOCK
    # ========================================================

    SP_block = (
        so_ds["so"]
        .isel(
            latitude=slice(
                hi0,
                hi1
            ),
            longitude=slice(
                hj0,
                hj1
            )
        )
        .values
    )

    theta_block = (
        theta_ds["thetao"]
        .isel(
            latitude=slice(
                hi0,
                hi1
            ),
            longitude=slice(
                hj0,
                hj1
            )
        )
        .values
    )

    ug_surface = (
        ssh_ds["ug"]
        .isel(
            latitude=slice(
                hi0,
                hi1
            ),
            longitude=slice(
                hj0,
                hj1
            )
        )
        .values
    )

    vg_surface = (
        ssh_ds["vg"]
        .isel(
            latitude=slice(
                hi0,
                hi1
            ),
            longitude=slice(
                hj0,
                hj1
            )
        )
        .values
    )

    # ========================================================
    # DYNAMIC HEIGHT
    # ========================================================

    D = compute_dynamic_height_block(
        SP_block,
        theta_block,
        lat,
        lon,
        depth
    )

    # ========================================================
    # HORIZONTAL GRADIENTS
    # ========================================================

    dDdy, dDdx = compute_horizontal_gradients(
        D,
        lat,
        lon
    )

    # ========================================================
    # CORIOLIS
    # ========================================================

    f = (
        2.0
        * OMEGA
        * np.sin(
            np.deg2rad(lat)
        )
    )

    f4d = f[
        None,
        None,
        :,
        None
    ]

    # ========================================================
    # RELATIVE GEOSTROPHIC VELOCITY
    # ========================================================

    ug_rel = -dDdy / f4d
    vg_rel =  dDdx / f4d

    # ========================================================
    # REMOVE SURFACE VALUE
    # ========================================================

    ug0 = ug_rel[:, 0, :, :]
    vg0 = vg_rel[:, 0, :, :]

    ug_rel = (
        ug_rel
        - ug0[:, None, :, :]
    )

    vg_rel = (
        vg_rel
        - vg0[:, None, :, :]
    )

    # ========================================================
    # ADD SSH SURFACE VELOCITY
    # ========================================================

    ug_geo = (
        ug_surface[:, None, :, :]
        + ug_rel
    )

    vg_geo = (
        vg_surface[:, None, :, :]
        + vg_rel
    )

    # ========================================================
    # SELECT TARGET DEPTHS
    # ========================================================

    ug_geo = ug_geo[
        :,
        depth_indices,
        :,
        :
    ]

    vg_geo = vg_geo[
        :,
        depth_indices,
        :,
        :
    ]

    # ========================================================
    # CLIMATOLOGY
    # ========================================================

    ug_clim = np.nanmean(
        ug_geo[clim_mask, :, :, :],
        axis=0
    )

    vg_clim = np.nanmean(
        vg_geo[clim_mask, :, :, :],
        axis=0
    )

    # ========================================================
    # ANOMALIES
    # ========================================================

    ug_anom = (
        ug_geo
        - ug_clim[None, :, :, :]
    )

    vg_anom = (
        vg_geo
        - vg_clim[None, :, :, :]
    )

    # ========================================================
    # EKE
    #
    # m2/s2 -> cm2/s2
    # ========================================================

    EKE = (
        0.5
        * (
            ug_anom**2
            +
            vg_anom**2
        )
        * 10000.0
    )

    # ========================================================
    # REMOVE HALO
    # ========================================================

    core_i0 = i0 - hi0
    core_i1 = core_i0 + (i1 - i0)

    core_j0 = j0 - hj0
    core_j1 = core_j0 + (j1 - j0)

    ug_geo = ug_geo[
        :,
        :,
        core_i0:core_i1,
        core_j0:core_j1
    ]

    vg_geo = vg_geo[
        :,
        :,
        core_i0:core_i1,
        core_j0:core_j1
    ]

    EKE = EKE[
        :,
        :,
        core_i0:core_i1,
        core_j0:core_j1
    ]

    # ========================================================
    # RETURN
    # ========================================================

    return (
        i0,
        i1,
        j0,
        j1,
        ug_geo.astype(np.float32),
        vg_geo.astype(np.float32),
        EKE.astype(np.float32)
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def compute_vertical_geo_EKE(
    so_file,
    thetao_file,
    ssh_file,
    output_file
):

    start = time.time()

    print("\n" + "=" * 70)
    print("VERTICAL GEOSTROPHIC EKE")
    print("=" * 70)

    # ========================================================
    # OPEN LAZILY
    # ========================================================

    print("\nOpening datasets lazily...")

    so_ds = xr.open_dataset(
        so_file
    )

    theta_ds = xr.open_dataset(
        thetao_file
    )

    ssh_ds = xr.open_dataset(
        ssh_file
    )

    # ========================================================
    # TIME SELECTION
    # ========================================================

    so_ds = so_ds.sel(
        time=slice(
            DATE_START,
            DATE_END
        )
    )

    theta_ds = theta_ds.sel(
        time=slice(
            DATE_START,
            DATE_END
        )
    )

    ssh_ds = ssh_ds.sel(
        time=slice(
            DATE_START,
            DATE_END
        )
    )



    # ========================================================
    # TEST MODE: REDUCE SPATIAL DOMAIN
    # ========================================================

    if TEST_MODE:

        print("\nTEST MODE ENABLED")

        # Keep the full time series for the climatology.
        # Reduce the spatial domain only.

        so_ds = so_ds.isel(
            latitude=slice(0, TEST_NLAT),
            longitude=slice(0, TEST_NLON)
        )

        theta_ds = theta_ds.isel(
            latitude=slice(0, TEST_NLAT),
            longitude=slice(0, TEST_NLON)
        )

        ssh_ds = ssh_ds.isel(
            latitude=slice(0, TEST_NLAT),
            longitude=slice(0, TEST_NLON)
        )

        print("SO:", so_ds.sizes)
        print("THETA:", theta_ds.sizes)
        print("SSH:", ssh_ds.sizes)









    # ========================================================
    # CHECK COORDINATES
    # ========================================================

    print("\nSO dimensions:")
    print(so_ds["so"].dims)
    print(so_ds["so"].shape)

    print("\nTHETA dimensions:")
    print(theta_ds["thetao"].dims)
    print(theta_ds["thetao"].shape)

    print("\nSSH velocity:")
    print(ssh_ds["ug"].dims)
    print(ssh_ds["ug"].shape)

    # ========================================================
    # DEPTH
    # ========================================================

    depth = so_ds.depth.values

    depth_indices, actual_depths = (
        find_depth_indices(
            depth,
            TARGET_DEPTHS
        )
    )

    print("\nRequested depths:")

    for target, actual in zip(
        TARGET_DEPTHS,
        actual_depths
    ):

        print(
            f"  {target:4d} m -> "
            f"{actual:.2f} m"
        )

    # ========================================================
    # TIME
    # ========================================================

    time_values = (
        so_ds.time.values
    )

    clim_mask = (
        (time_values >= np.datetime64(CLIM_START))
        &
        (time_values <= np.datetime64(CLIM_END))
    )

    print(
        "\nClimatology:",
        CLIM_START,
        "->",
        CLIM_END
    )

    print(
        "Climatology points:",
        clim_mask.sum()
    )



    # ========================================================
    # TEST MODE: SELECT OUTPUT DATES
    # ========================================================

    if TEST_MODE:

        test_times = so_ds.time.values[:TEST_NTIME]

        print(
            "\nOutput dates for test:",
            test_times
        )










    # ========================================================
    # GRID
    # ========================================================

    nlat = so_ds.sizes["latitude"]
    nlon = so_ds.sizes["longitude"]
    nt = so_ds.sizes["time"]
    nd = len(TARGET_DEPTHS)

    print(
        f"\nGrid = {nlat} x {nlon}"
    )

    print(
        f"Time = {nt}"
    )

    print(
        f"Depths = {nd}"
    )

    print(
        f"Block = "
        f"{BLOCK_LAT} x {BLOCK_LON}"
    )

    print(
        f"N_JOBS = {N_JOBS}"
    )

    # ========================================================
    # CREATE OUTPUT DATASET
    # ========================================================

    coords = {
        "time": so_ds.time,
        "depth": actual_depths,
        "latitude": so_ds.latitude,
        "longitude": so_ds.longitude
    }

    # --------------------------------------------------------
    # We create an empty file first.
    # --------------------------------------------------------

    empty = np.empty(
        (
            nt,
            nd,
            nlat,
            nlon
        ),
        dtype=np.float32
    )

    empty[:] = np.nan

    output_ds = xr.Dataset(
        {
            "ug": (
                (
                    "time",
                    "depth",
                    "latitude",
                    "longitude"
                ),
                empty.copy()
            ),

            "vg": (
                (
                    "time",
                    "depth",
                    "latitude",
                    "longitude"
                ),
                empty.copy()
            ),

            "EKE_geo": (
                (
                    "time",
                    "depth",
                    "latitude",
                    "longitude"
                ),
                empty.copy()
            )
        },
        coords=coords
    )

    output_ds["ug"].attrs = {
        "long_name":
            "Absolute geostrophic zonal velocity",
        "description":
            "SSH surface velocity + T/S vertical shear",
        "units":
            "m s-1"
    }

    output_ds["vg"].attrs = {
        "long_name":
            "Absolute geostrophic meridional velocity",
        "description":
            "SSH surface velocity + T/S vertical shear",
        "units":
            "m s-1"
    }

    output_ds["EKE_geo"].attrs = {
        "long_name":
            "Geostrophic eddy kinetic energy",
        "description":
            "Velocity anomalies relative to 1993-2012 climatology",
        "units":
            "cm2 s-2"
    }

    # ========================================================
    # WRITE INITIAL FILE
    # ========================================================

    output_ds.to_netcdf(
        output_file,
        mode="w"
    )

    del output_ds
    del empty

    # ========================================================
    # NUMBER OF BLOCKS
    # ========================================================

    nblocks_lat = int(
        np.ceil(
            nlat / BLOCK_LAT
        )
    )

    nblocks_lon = int(
        np.ceil(
            nlon / BLOCK_LON
        )
    )

    total_blocks = (
        nblocks_lat
        * nblocks_lon
    )

    block_number = 0

    # ========================================================
    # PROCESS BLOCKS
    #
    # IMPORTANT:
    #
    # We DO NOT launch all blocks simultaneously.
    #
    # Only ONE block is loaded at a time.
    #
    # Parallelization occurs inside the block on the
    # independent vertical profiles.
    # ========================================================

    for i0 in range(
        0,
        nlat,
        BLOCK_LAT
    ):

        i1 = min(
            i0 + BLOCK_LAT,
            nlat
        )

        for j0 in range(
            0,
            nlon,
            BLOCK_LON
        ):

            j1 = min(
                j0 + BLOCK_LON,
                nlon
            )

            block_number += 1

            print(
                "\n" + "-" * 70
            )

            print(
                f"BLOCK "
                f"{block_number}/{total_blocks}"
            )

            print(
                f"lat={i0}:{i1}, "
                f"lon={j0}:{j1}"
            )

            # =================================================
            # PROCESS BLOCK
            # =================================================

            (
                bi0,
                bi1,
                bj0,
                bj1,
                ug_block,
                vg_block,
                EKE_block
            ) = process_block(
                so_ds,
                theta_ds,
                ssh_ds,
                i0,
                i1,
                j0,
                j1,
                depth,
                depth_indices,
                clim_mask
            )

            # =================================================
            # WRITE ONLY THIS BLOCK
            # =================================================

            block_ds = xr.Dataset(
                {
                    "ug": (
                        (
                            "time",
                            "depth",
                            "latitude",
                            "longitude"
                        ),
                        ug_block
                    ),

                    "vg": (
                        (
                            "time",
                            "depth",
                            "latitude",
                            "longitude"
                        ),
                        vg_block
                    ),

                    "EKE_geo": (
                        (
                            "time",
                            "depth",
                            "latitude",
                            "longitude"
                        ),
                        EKE_block
                    )
                },

                coords={
                    "time":
                        so_ds.time.values,

                    "depth":
                        actual_depths,

                    "latitude":
                        so_ds.latitude.values[
                            bi0:bi1
                        ],

                    "longitude":
                        so_ds.longitude.values[
                            bj0:bj1
                        ]
                }
            )

            with Dataset(output_file, "a") as nc:
                nc.variables["ug"][:, :, bi0:bi1, bj0:bj1] = ug_block
                nc.variables["vg"][:, :, bi0:bi1, bj0:bj1] = vg_block
                nc.variables["EKE_geo"][:, :, bi0:bi1, bj0:bj1] = EKE_block

            # =================================================
            # FREE MEMORY
            # =================================================

            del block_ds
            del ug_block
            del vg_block
            del EKE_block

            print(
                f"Block {block_number} written."
            )

    # ========================================================
    # CLOSE
    # ========================================================

    so_ds.close()
    theta_ds.close()
    ssh_ds.close()

    elapsed = (
        time.time()
        - start
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "DONE"
    )

    print(
        "Output:",
        output_file
    )

    print(
        f"Elapsed time: "
        f"{elapsed / 60:.1f} min"
    )

    print(
        "=" * 70
    )


# ============================================================
# ALBORAN
# ============================================================

# compute_vertical_geo_EKE(

#     so_file="../../data/006_004_medmfc_so_alb.nc",

#     thetao_file="../../data/006_004_medmfc_thetao_alb.nc",

#     ssh_file="../../results/006_004_medmfc_vg_alb.nc",

#     output_file=(
#         "../../results/"
#         "006_004_medmfc_vertical_geo_EKE_alb.nc"
#     )
# )



# ============================================================
# IERAPETRA
# ============================================================

compute_vertical_geo_EKE(

    so_file="../../data/006_004_medmfc_so_iera.nc",

    thetao_file="../../data/006_004_medmfc_thetao_iera.nc",

    ssh_file="../../results/006_004_medmfc_vg_iera.nc",

    output_file=(
        "../../results/"
        "006_004_medmfc_vertical_geo_EKE_iera.nc"
    )
)