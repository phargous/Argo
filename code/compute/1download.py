import copernicusmarine
import numpy as np
import xarray as xr
import os
import sys
sys.path.append('../')
import time
from datetime import datetime, timedelta

from unittest.mock import patch



# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = "../../data/"

LON_MIN = -5.4
LON_MAX = -1
LAT_MIN = 35
LAT_MAX = 37

DATE_START = "1993-01-01T00:00:00"
DATE_END = "2026-08-31T23:59:59"

DEPTH_MIN = 1.0182366371154785
DEPTH_MAX = 5754.0439453125

SUBSET_PARAMS = {
    "minimum_longitude": LON_MIN,
    "maximum_longitude": LON_MAX,
    "minimum_latitude": LAT_MIN,
    "maximum_latitude": LAT_MAX,
    "start_datetime": DATE_START,
    "end_datetime": DATE_END,
    "minimum_depth": DEPTH_MIN,
    "maximum_depth": DEPTH_MAX,
    "output_directory": OUTPUT_DIR,
    "force_download": True,
}




# ============================================================
# Datasets to download
# ============================================================

datasets = [

    {
        "dataset_id": "cmems_mod_med_phy-cur_my_4.2km_P1D-m",
        "variables": ["uo", "vo"],
        "output_filename": "006_004_medmfc_uovo.nc",
    },

    {
        "dataset_id": "cmems_mod_med_phy-sal_my_4.2km_P1D-m",
        "variables": ["so"],
        "output_filename": "006_004_medmfc_so.nc",
    },

    {
        "dataset_id": "cmems_mod_med_phy-temp_my_4.2km_P1D-m",
        "variables": ["thetao"],
        "output_filename": "006_004_medmfc_thetao.nc",
    },

    {
        "dataset_id": "cmems_mod_med_phy-hflux_my_4.2km_P1D-m",
        "variables": ["hfds", "hfls", "hfss", "rlds", "rsntds"],
        "output_filename": "006_004_medmfc_hflux.nc",
    },

    {
        "dataset_id": "cmems_mod_med_phy-mld_my_4.2km_P1D-m",
        "variables": ["mlotst"],
        "output_filename": "006_004_medmfc_mld.nc",
    },

    {
        "dataset_id": "cmems_mod_med_phy-mflux_my_4.2km_P1D-m",
        "variables": ["tauuo", "tauvo"],
        "output_filename": "006_004_medmfc_mflux.nc",
    },

    {
        "dataset_id": "cmems_mod_med_phy-ssh_my_4.2km_P1D-m",
        "variables": ["zos"],
        "output_filename": "006_004_medmfc_ssh.nc",
    },

    {
        "dataset_id": "cmems_mod_med_phy_my_4.2km_static",
        "variables": ["deptho", "deptho_lev", "mask"],
        "output_filename": "006_004_medmfc_bathy.nc",
    },
]


# ============================================================
# Download
# ============================================================

for dataset in datasets:

    print(f"\nDownloading: {dataset['output_filename']}")

    copernicusmarine.subset(
        dataset_id=dataset["dataset_id"],
        variables=dataset["variables"],
        output_filename=dataset["output_filename"],
        **SUBSET_PARAMS,
    )

    print(f"Finished: {dataset['output_filename']}")