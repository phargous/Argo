#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ierapetra (modèle MedMFC seul) : forçage dans le lobe de pompage, stratification, énergie
potentielle disponible (APE) et vitesses verticales, par saison (DJF, MAM, JJA, SON).

  A. EKE du modèle (ug, vg) -> région "eddy" (isoligne EKE >= EKE_ISOLINE) et région
     "control" (EKE moyenne faible).
  B. Pompage d'Ekman w_E -> région "lobe" = zone connexe où w_E (climatologie JJA) <= LOBE_THRESHOLD ;
     indicateurs de forçage dans le lobe et dans la région eddy.
  C. Stratification + dynamique (so, thetao, uo, vo ; 0-MAX_DEPTH m) sans jamais charger les
     fichiers en entier : blocs de temps x tuiles de 10x10 mailles.
       - N2, T, S par couche, profondeur des isopycnes, maximum de salinité (LIW)
       - w par continuité (intégrée depuis la surface, w(0)=0, >0 vers le haut)
       - APE (QG) relative au profil moyen de la région "control" à la même date
       - conversion de flottabilité APE -> EKE : -g <rho' w'>
       - cartes, coupe à SECTION_LAT, histogrammes T-S
  D. Statistiques (P1 vs P2, corrélations décalées avec l'EKE de SON) et figures.

Définitions :
  w_E     = (1/rho0) k.curl(tau/f)     (m/jour ; >0 succion, <0 pompage)
  w_down  = moyenne spatiale de max(-w_E, 0) : grandeur (positive) du pompage descendant
  N2      = (g/rho0) d(sigma0)/dz     (z positif vers le bas)
  APE     = g^2 rho'^2 / (2 rho0 N2_ref)   (J/m3), intégrée par couche (J/m2) ;
            rho' = sigma0 - profil moyen "control" ; N2_ref plancher N2_FLOOR
  conv    = -g rho' w'  (W/m3), intégrée par couche (W/m2) ; >0 : APE -> EKE ;
            w' = w - w moyen "control"

Dépendances : numpy, pandas, scipy, matplotlib, xarray, gsw.
Lancer :  python ierapetra_forcage_stratification.py
"""
import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy import ndimage, stats, signal
from scipy.integrate import cumulative_trapezoid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =============================== CONFIG ====================================
VG_FILE = "../../results/006_004_medmfc_vg_iera.nc"
UG_VAR, VG_VAR = "ug", "vg"
VG_FACTOR = 1.0                     # 0.01 si les vitesses sont en cm/s
DATA = "../../data/"
MFLUX_FILE = DATA + "006_004_medmfc_mflux_iera.nc"
MLD_FILE = DATA + "006_004_medmfc_mld_iera.nc"
THETAO_FILE = DATA + "006_004_medmfc_thetao_iera.nc"
SO_FILE = DATA + "006_004_medmfc_so_iera.nc"
UOVO_FILE = DATA + "006_004_medmfc_uovo_iera.nc"

MODEL_TAUX, MODEL_TAUY = None, None  # None = auto (standard_name / noms usuels)
MODEL_SIGN = 1.0                    # -1 si le contrôle de signe échoue
MLD_VAR = None                      # None = auto (mlotst, mld, sinon 1re variable)
UO_VAR, VO_VAR = None, None         # None = auto (standard_name / uo, vo)

START, END = "1993-01-01", "2025-12-31"   # les fichiers vont jusqu'en 2026
PERIOD1 = ("1993", "2012")
PERIOD2 = ("2013", "2025")

# --- régions ---
EKE_ISOLINE = 300.0                 # cm2/s2 : région eddy (EKE moyenne >= seuil)
BOX_PERIOD = (START, END)
EKE_REF_PERIOD = (START, END)
CONTROL_EKE = 150.0                 # région témoin : EKE moyenne < CONTROL_EKE
LOBE_SEASON = "JJA"                 # saison de la climatologie de w_E définissant le lobe
LOBE_THRESHOLD = -0.5               # m/jour : lobe = zone connexe où w_E <= seuil
LOBE_BOX = dict(lon=(24.5, 28.0), lat=(33.0, 35.0))   # recherche limitée au sud de la Crète

# --- saisons ---
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
MET_SEASONS = tuple(SEASONS)
EKE_SEASON = "SON"                  # EKE de référence (corrélations)
STRONG_SEASON = "JJA"               # saison fixant le seuil des "jours forts"

# --- forçage ---
SMOOTH_DEG_MODEL = 0.25
EDGE_TRIM = 2
STRONG_PCT = 90

# --- stratification / dynamique ---
MAX_DEPTH = 1000.0                  # m
TILE = 10                           # tuiles de 10x10 mailles
TIME_CHUNK = 30                     # dates lues à la fois
TIME_STRIDE = 5                     # 1 = toutes les dates (long) ; 5 = une sur 5 (rapide)
LAYERS = ((0, 100), (100, 300), (300, 700), (700, 1000))
ISOPYCNALS = (27.5, 28.5, 29.0)
LIW_RANGE = (150.0, 700.0)
DO_DYNAMICS = True                  # w, APE, conversion (lit uo, vo ; plus long)
N2_FLOOR = 1e-6                     # s-2 : plancher de N2_ref dans l'APE
RECOMPUTE_STRAT = True              # False : relire out/strat_results.pkl

# --- coupe verticale et T-S ---
SECTION_LAT = 34.25
SECTION_ZMAX = 700.0
SECTION_ISOPYCNALS = np.arange(26.0, 29.5, 0.25)
SECTION_LABEL_FS = 11
TS_T = (12.0, 30.0, 0.1)
TS_S = (38.2, 39.8, 0.01)

OUTDIR = "out_ierapetra"
RHO0, G = 1025.0, 9.81
OMEGA, R_EARTH = 7.2921e-5, 6.371e6
VERSION = "v5 : 4 saisons, lobe de pompage, APE, w par continuite, correlations decalees"


# ============================ SAISONS =======================================
def season_year(idx, months):
    """Année de la saison : pour DJF, décembre compte pour l'année suivante."""
    yr = np.asarray(idx.year).copy()
    if 12 in months and (1 in months or 2 in months):
        yr = yr + (np.asarray(idx.month) == 12)
    return yr


def season_mask(idx, months, period):
    yr = season_year(idx, months)
    return np.asarray(idx.month.isin(months)) & (yr >= int(period[0])) & (yr <= int(period[1]))


def seasonal_mean(s, months):
    """Moyenne par saison-année ; les saisons incomplètes (mois manquants) sont écartées."""
    s = s.dropna()
    s = s[s.index.month.isin(months)]
    if s.empty:
        return pd.Series(dtype=float)
    yr = season_year(s.index, months)
    out = s.groupby(yr).mean()
    nm = pd.Series(np.asarray(s.index.month), index=s.index).groupby(yr).nunique()
    return out[nm >= len(months)]


# ============================ NOYAU NUMÉRIQUE ===============================
def nan_gaussian(arr, sigma_y, sigma_x):
    """Filtre gaussien normalisé sur (t, y, x), tolérant aux NaN."""
    if sigma_y <= 0 and sigma_x <= 0:
        return arr
    ok = np.isfinite(arr)
    sig = (0, sigma_y, sigma_x)
    num = ndimage.gaussian_filter(np.where(ok, arr, 0.0), sigma=sig, mode="nearest")
    den = ndimage.gaussian_filter(ok.astype(float), sigma=sig, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[(den < 0.5) | ~ok] = np.nan
    return out


def ekman_pumping_np(taux, tauy, lon, lat):
    """w_E (m/s, >0 vers le haut) = (1/rho0) k.curl(tau/f), rotationnel sphérique exact.
    taux, tauy : (t, y, x) en N m-2 ; lon, lat en degrés croissants."""
    phi = np.deg2rad(lat)
    c = np.cos(phi)[None, :, None]
    f = (2.0 * OMEGA * np.sin(phi))[None, :, None]
    dAy_dlam = np.gradient(tauy / f, np.deg2rad(lon), axis=2)
    dAxc_dphi = np.gradient(taux / f * c, phi, axis=1)
    return (dAy_dlam - dAxc_dphi) / (RHO0 * R_EARTH * c)


def vertical_velocity(u, v, lon, lat, depth):
    """w (m/s, >0 vers le haut) par continuité, intégrée depuis la surface avec w(0) = 0.
    u, v : (t, z, y, x) en m/s ; depth (z,) m positifs vers le bas.
    D = [d(u)/dlam + d(v cos phi)/dphi] / (R cos phi) ;  w(d) = integrale_0^d D dd'."""
    phi = np.deg2rad(lat)
    c = np.cos(phi)
    dudl = np.gradient(u, np.deg2rad(lon), axis=3)
    dvcdp = np.gradient(v * c[None, None, :, None], phi, axis=2)
    div = (dudl + dvcdp) / (R_EARTH * c)[None, None, :, None]
    w = cumulative_trapezoid(div, depth, axis=1, initial=0.0) + div[:, :1] * depth[0]
    return w.astype("float32")


def region_mean(a, mask, lat):
    """Moyenne pondérée cos(lat) de a (t,y,x) sur mask (y,x), tolérante aux NaN."""
    wm = mask.astype(float) * np.cos(np.deg2rad(lat))[:, None]
    fin = np.isfinite(a)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.where(fin, a, 0.0) * wm).sum(axis=(1, 2)) / (fin * wm).sum(axis=(1, 2))


def _component(cond, field, pick_max):
    lab, _ = ndimage.label(cond, structure=np.ones((3, 3)))
    key = np.where(cond, field, -np.inf if pick_max else np.inf)
    i = np.unravel_index(np.nanargmax(key) if pick_max else np.nanargmin(key), field.shape)
    return lab == lab[i]


def largest_region_at_max(eke_map, threshold):
    """Composante connexe de {EKE >= threshold} contenant le maximum d'EKE."""
    cond = np.isfinite(eke_map) & (eke_map >= threshold)
    if not cond.any():
        raise ValueError(f"Aucune maille avec EKE >= {threshold} (max = "
                         f"{np.nanmax(eke_map):.0f} cm2/s2). Baisser EKE_ISOLINE.")
    return _component(cond, eke_map, True)


def box_mask(lon, lat, box):
    return ((lat >= box["lat"][0]) & (lat <= box["lat"][1]))[:, None] & \
        ((lon >= box["lon"][0]) & (lon <= box["lon"][1]))[None, :]


def build_lobe(w_clim, lon, lat):
    """Lobe de pompage : composante connexe de {w_E <= LOBE_THRESHOLD} (dans LOBE_BOX)
    contenant le minimum de w_E."""
    bm = box_mask(lon, lat, LOBE_BOX)
    cond = np.isfinite(w_clim) & (w_clim <= LOBE_THRESHOLD) & bm
    if not cond.any():
        mn = np.nanmin(np.where(bm, w_clim, np.nan))
        raise ValueError(f"Aucune maille avec w_E <= {LOBE_THRESHOLD} m/jour ({LOBE_SEASON}) dans "
                         f"LOBE_BOX (min = {mn:.2f}). Relever LOBE_THRESHOLD.")
    return _component(cond, w_clim, False)


def check_grid(lon_a, lat_a, lon_b, lat_b, name):
    if len(lon_a) != len(lon_b) or len(lat_a) != len(lat_b) or \
            not (np.allclose(lon_a, lon_b, atol=1e-3) and np.allclose(lat_a, lat_b, atol=1e-3)):
        raise ValueError(f"Grille de '{name}' différente de celle des vitesses : à réinterpoler.")


# ======================== STRATIFICATION (numpy pur) ========================
def compute_sigma0(sal, theta, depth, lat, lon):
    """sigma0 (kg/m3) depuis salinité pratique et température potentielle.
    sal, theta : (t, z, y, x) ; depth (z,) m positifs ; lat (y,), lon (x,)."""
    import gsw
    shp = sal.shape
    p = np.broadcast_to(gsw.p_from_z(-depth, float(np.mean(lat)))[None, :, None, None], shp)
    la = np.broadcast_to(lat[None, None, :, None], shp)
    lo = np.broadcast_to(lon[None, None, None, :], shp)
    sa = gsw.SA_from_SP(sal, p, lo, la)
    ct = gsw.CT_from_pt(sa, theta)
    return gsw.sigma0(sa, ct)


def isopycnal_depth(sig0, depth, target):
    """Profondeur (m) de la 1re traversée de sigma0 = target depuis la surface (interpolation
    linéaire). NaN si absente ou déjà dépassée en surface."""
    ge = sig0 >= target
    has = ge.any(axis=1)
    k = ge.argmax(axis=1)
    k0 = np.clip(k - 1, 0, None)
    s1 = np.take_along_axis(sig0, k[:, None], 1)[:, 0]
    s0 = np.take_along_axis(sig0, k0[:, None], 1)[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = (target - s0) / (s1 - s0)
        z = depth[k0] + frac * (depth[k] - depth[k0])
    return np.where(has & (k > 0) & np.isfinite(z), z, np.nan)


def strat_names():
    names = []
    for z1, z2 in LAYERS:
        names += [f"N2_{z1}-{z2}m", f"T_{z1}-{z2}m", f"S_{z1}-{z2}m"]
    names += [f"z_sig{s}" for s in ISOPYCNALS]
    return names + ["z_Smax", "Smax"]


def dyn_names():
    return [f"{p}_{z1}-{z2}m" for z1, z2 in LAYERS for p in ("APE", "conv", "w", "absw")]


def diag_names():
    return strat_names() + (dyn_names() if DO_DYNAMICS else [])


def strat_diagnostics(theta, sal, depth, lat, lon, return_sig=False):
    """theta, sal : (t, z, y, x) float64 -> dict nom -> (t, y, x) [, sigma0].
    Les moyennes par couche exigent que le niveau le plus profond de la couche soit valide."""
    sig0 = compute_sigma0(sal, theta, depth, lat, lon)
    n2 = (G / RHO0) * np.gradient(sig0, depth, axis=1)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for z1, z2 in LAYERS:
            ks = np.where((depth >= z1) & (depth < z2))[0]
            if len(ks) == 0:
                continue
            full = np.isfinite(sal[:, ks[-1]]) & np.isfinite(n2[:, ks[-1]])
            for name, f3 in ((f"N2_{z1}-{z2}m", n2), (f"T_{z1}-{z2}m", theta),
                             (f"S_{z1}-{z2}m", sal)):
                out[name] = np.where(full, np.nanmean(f3[:, ks], axis=1), np.nan)
        for s in ISOPYCNALS:
            out[f"z_sig{s}"] = isopycnal_depth(sig0, depth, s)
        ks = np.where((depth >= LIW_RANGE[0]) & (depth <= LIW_RANGE[1]))[0]
        s_liw = np.where(np.isfinite(sal[:, ks]), sal[:, ks], -np.inf)
        i = s_liw.argmax(axis=1)
        smax = np.take_along_axis(s_liw, i[:, None], 1)[:, 0]
        zmax = depth[ks][i]
        bad = ~np.isfinite(smax)
        out["Smax"] = np.where(bad, np.nan, smax)
        out["z_Smax"] = np.where(bad, np.nan, zmax)
    return (out, sig0) if return_sig else out


def reference_n2(sig_ref, depth):
    """N2 du profil de référence (t, z), borné inférieurement par N2_FLOOR."""
    return np.maximum((G / RHO0) * np.gradient(sig_ref, depth, axis=1), N2_FLOOR)


def dynamics_diagnostics(sig0, w, sig_ref, w_ref, n2_ref, depth, dz):
    """APE, conversion de flottabilité, w moyen et |w| moyen par couche.
    sig0, w : (t, z, y, x) ; sig_ref, w_ref, n2_ref : (t, z) -> dict nom -> (t, y, x)."""
    rho_p = sig0 - sig_ref[:, :, None, None]
    w_p = w - w_ref[:, :, None, None]
    ape = G ** 2 * rho_p ** 2 / (2.0 * RHO0 * n2_ref[:, :, None, None])     # J/m3
    conv = -G * rho_p * w_p                                                  # W/m3
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for z1, z2 in LAYERS:
            ks = np.where((depth >= z1) & (depth < z2))[0]
            if len(ks) == 0:
                continue
            full = np.isfinite(sig0[:, ks[-1]]) & np.isfinite(w[:, ks[-1]])
            wl = dz[ks][None, :, None, None]
            out[f"APE_{z1}-{z2}m"] = np.where(full, np.nansum(ape[:, ks] * wl, axis=1), np.nan)
            out[f"conv_{z1}-{z2}m"] = np.where(full, np.nansum(conv[:, ks] * wl, axis=1), np.nan)
            out[f"w_{z1}-{z2}m"] = np.where(full, np.nanmean(w[:, ks], axis=1) * 86400.0, np.nan)
            out[f"absw_{z1}-{z2}m"] = np.where(full, np.nanmean(np.abs(w[:, ks]), axis=1) * 86400.0, np.nan)
    return out


def layer_thickness(depth):
    """Épaisseur (m) associée à chaque niveau (limites à mi-distance)."""
    mid = 0.5 * (depth[1:] + depth[:-1])
    edges = np.r_[0.0, mid, depth[-1] + 0.5 * (depth[-1] - depth[-2])]
    return np.diff(edges)


def ts_axes():
    t0, t1, dt = TS_T
    s0, s1, ds = TS_S
    return (t0 + dt * np.arange(int(round((t1 - t0) / dt)) + 1),
            s0 + ds * np.arange(int(round((s1 - s0) / ds)) + 1))


def add_ts(hist, T, S, dz):
    """Ajoute à l'histogramme T-S (aplati) les points T,S (t',z,n) pondérés par dz (z,)."""
    nT, nS = len(ts_axes()[0]) - 1, len(ts_axes()[1]) - 1
    ok = np.isfinite(T) & np.isfinite(S)
    w = np.broadcast_to(dz[None, :, None], T.shape)
    ti = np.where(ok, np.floor((T - TS_T[0]) / TS_T[2]), -1).astype(np.int64)
    si = np.where(ok, np.floor((S - TS_S[0]) / TS_S[2]), -1).astype(np.int64)
    v = (ti >= 0) & (ti < nT) & (si >= 0) & (si < nS)
    hist += np.bincount(ti[v] * nS + si[v], weights=w[v], minlength=nT * nS)


# ================================ LECTURE ===================================
def _std_coords(ds):
    ren = {k: v for k, v in {"valid_time": "time", "lat": "latitude",
                             "lon": "longitude"}.items() if k in ds.variables or k in ds.dims}
    return ds.rename(ren) if ren else ds


def _open2d(path):
    import xarray as xr
    ds = _std_coords(xr.open_dataset(path))
    return ds.sortby("latitude").sortby("longitude")


def _pack2d(ds, variables, sign=1.0):
    ds = ds.sel(time=slice(START, END)).transpose("time", "latitude", "longitude").load()
    out = dict(time=pd.DatetimeIndex(ds["time"].values),
               lon=ds["longitude"].values.astype(float), lat=ds["latitude"].values.astype(float))
    for v in variables:
        out[v] = ds[v].values.astype("float32") * (sign if v in ("taux", "tauy") else 1.0)
    return out


def load_vg():
    ds = _open2d(VG_FILE)
    ds = ds[[UG_VAR, VG_VAR]].squeeze(drop=True).rename({UG_VAR: "ug", VG_VAR: "vg"})
    d = _pack2d(ds, ["ug", "vg"])
    d["ug"] = d["ug"] * np.float32(VG_FACTOR)
    d["vg"] = d["vg"] * np.float32(VG_FACTOR)
    print(f"[vg] vitesse max : {np.nanmax(np.hypot(d['ug'], d['vg'])):.2f} m/s "
          "(si > 5, vérifier VG_FACTOR : vitesses en cm/s ?)")
    return d


def _find_stress_vars(ds):
    vx, vy = MODEL_TAUX, MODEL_TAUY
    for v in ds.data_vars:
        sn = ds[v].attrs.get("standard_name", "").lower()
        if "stress" in sn and ("eastward" in sn or "x_stress" in sn):
            vx = vx or v
        if "stress" in sn and ("northward" in sn or "y_stress" in sn):
            vy = vy or v
    low = {v.lower(): v for v in ds.data_vars}
    vx = vx or next((low[n] for n in ("tauuo", "sozotaux", "utau", "taux", "sustr") if n in low), None)
    vy = vy or next((low[n] for n in ("tauvo", "sometauy", "vtau", "tauy", "svstr") if n in low), None)
    if not (vx and vy):
        raise KeyError(f"Stress introuvable. Variables : {list(ds.data_vars)}. "
                       "Renseigner MODEL_TAUX / MODEL_TAUY.")
    return vx, vy


def load_mflux():
    ds = _open2d(MFLUX_FILE)
    vx, vy = _find_stress_vars(ds)
    print(f"[mflux] variables : {vx}, {vy}")
    ds = ds[[vx, vy]].squeeze(drop=True).rename({vx: "taux", vy: "tauy"})
    return _pack2d(ds, ["taux", "tauy"], MODEL_SIGN)


def load_mld():
    ds = _open2d(MLD_FILE)
    v = MLD_VAR or next((n for n in ("mlotst", "mld", "mlp") if n in ds.variables),
                        list(ds.data_vars)[0])
    ds = ds[[v]].squeeze(drop=True).rename({v: "mld"})
    return _pack2d(ds, ["mld"])


def _find_uv_vars(ds):
    vu, vv = UO_VAR, VO_VAR
    for v in ds.data_vars:
        sn = ds[v].attrs.get("standard_name", "").lower()
        if "eastward" in sn and "velocity" in sn:
            vu = vu or v
        if "northward" in sn and "velocity" in sn:
            vv = vv or v
    low = {v.lower(): v for v in ds.data_vars}
    vu = vu or low.get("uo")
    vv = vv or low.get("vo")
    if not (vu and vv):
        raise KeyError(f"uo/vo introuvables. Variables : {list(ds.data_vars)}. "
                       "Renseigner UO_VAR / VO_VAR.")
    return vu, vv


# def open3d():
#     """Ouvre (sans rien charger) thetao, so et (si DO_DYNAMICS) uo/vo ; vérifie les axes."""
#     import xarray as xr
#     dt = _std_coords(xr.open_dataset(THETAO_FILE))
#     ds = _std_coords(xr.open_dataset(SO_FILE))
#     duv = None
#     if DO_DYNAMICS:
#         duv = _std_coords(xr.open_dataset(UOVO_FILE))
#         vu, vv = _find_uv_vars(duv)
#         print(f"[uovo] variables : {vu}, {vv}")
#         duv = duv.rename({vu: "uo", vv: "vo"})
#     for name, other in (("so", ds), ("uovo", duv)):
#         if other is None:
#             continue
#         for c in ("time", "depth", "latitude", "longitude"):
#             if c not in other.coords or not np.array_equal(dt[c].values, other[c].values):
#                 raise ValueError(f"Axe '{c}' différent entre thetao et {name}.")
#     for c in ("latitude", "longitude", "depth"):
#         if not np.all(np.diff(dt[c].values) > 0):
#             raise ValueError(f"'{c}' doit être croissant (trier les fichiers).")
#     return dt, ds, duv

def open3d():
    """Ouvre thetao, so et (si DO_DYNAMICS) uo/vo. Si nécessaire, coupe uovo pour conserver 
    les profondeurs communes. Vérifie ensuite la cohérence des axes."""
    import xarray as xr
    dt = _std_coords(xr.open_dataset(THETAO_FILE))
    ds = _std_coords(xr.open_dataset(SO_FILE))
    duv = None
    if DO_DYNAMICS:
        duv = _std_coords(xr.open_dataset(UOVO_FILE))
        vu, vv = _find_uv_vars(duv)
        print(f"[uovo] variables : {vu}, {vv}")
        duv = duv.rename({vu: "uo", vv: "vo"})
        for c in ("time", "latitude", "longitude"):
            if c not in duv.coords or not np.array_equal(dt[c].values, duv[c].values):
                raise ValueError(f"Axe '{c}' différent entre thetao et uovo.")
        depth_thetao = dt["depth"].values
        depth_uovo = duv["depth"].values
        if not np.array_equal(depth_thetao, depth_uovo):
            print("[uovo] Axe depth différent : sélection des profondeurs communes.")
            common_depths = np.intersect1d(depth_thetao, depth_uovo)
            if len(common_depths) == 0:
                raise ValueError("Aucune profondeur commune entre thetao et uovo.")
            duv = duv.sel(depth=common_depths)
            print(f"[uovo] Profondeurs conservées : {len(common_depths)} niveaux, "
                f"{common_depths[0]} à {common_depths[-1]} m.")
    for name, other in (("so", ds), ("uovo", duv)):
        if other is None:
            continue
        for c in ("time", "depth", "latitude", "longitude"):
            if c not in other.coords or not np.array_equal(dt[c].values, other[c].values):
                raise ValueError(f"Axe '{c}' différent entre thetao et {name}.")
    for c in ("latitude", "longitude", "depth"):
        if not np.all(np.diff(dt[c].values) > 0):
            raise ValueError(f"'{c}' doit être croissant (trier les fichiers).")
    return dt, ds, duv


# ======================= A. EKE ET RÉGIONS (par isoligne) ===================
def compute_eke(vel):
    """EKE (cm2/s2) en (t, y, x) float32 depuis ug, vg (m/s), par blocs annuels, et carte
    moyenne sur BOX_PERIOD. Anomalies relatives à EKE_REF_PERIOD."""
    t, u, v = vel["time"], vel["ug"], vel["vg"]
    ref = (t >= pd.Timestamp(EKE_REF_PERIOD[0])) & (t <= pd.Timestamp(EKE_REF_PERIOD[1]))
    years = np.unique(t.year)
    su = sv = n = 0.0
    for y in years:
        sel = np.where((t.year == y) & ref)[0]
        if len(sel):
            su = su + np.nansum(u[sel], 0, dtype="float64")
            sv = sv + np.nansum(v[sel], 0, dtype="float64")
            n = n + np.isfinite(u[sel]).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        um, vm = su / n, sv / n
    eke = np.full(u.shape, np.nan, dtype="float32")
    for y in years:
        sel = np.where(t.year == y)[0]
        eke[sel] = (0.5 * ((u[sel] - um) ** 2 + (v[sel] - vm) ** 2) * 1e4).astype("float32")
    per = (t >= pd.Timestamp(BOX_PERIOD[0])) & (t <= pd.Timestamp(BOX_PERIOD[1]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        eke_map = np.nanmean(eke[per], axis=0)
    return eke, eke_map


def build_masks(eke_map):
    eddy = largest_region_at_max(eke_map, EKE_ISOLINE)
    control = np.isfinite(eke_map) & (eke_map < CONTROL_EKE)
    print(f"[régions] eddy : {eddy.sum()} mailles (EKE >= {EKE_ISOLINE:.0f}) ; "
          f"control : {control.sum()} mailles (EKE < {CONTROL_EKE:.0f})")
    return {"eddy": eddy, "control": control}


# ============================ B. FORÇAGE ====================================
def ekman_fields(mflux):
    """w_E (t,y,x en m/jour) et stress lissés (N m-2), float32."""
    time, lon, lat = mflux["time"], mflux["lon"], mflux["lat"]
    taux, tauy = mflux["taux"], mflux["tauy"]
    m = np.asarray(time.month.isin(SEASONS["JJA"]))
    print(f"[contrôle] tau_y moyen JJA : {np.nanmean(tauy[m]):+.4f} N/m2 "
          "(attendu < 0 : Étésiens de nord). Sinon MODEL_SIGN = -1.")
    dlon, dlat = np.abs(np.diff(lon)).mean(), np.abs(np.diff(lat)).mean()
    sy, sx = SMOOTH_DEG_MODEL / dlat, SMOOTH_DEG_MODEL / dlon
    taux_s = nan_gaussian(taux, sy, sx).astype("float32")
    tauy_s = nan_gaussian(tauy, sy, sx).astype("float32")
    w = (ekman_pumping_np(taux_s.astype("float64"), tauy_s.astype("float64"), lon, lat)
         * 86400.0).astype("float32")
    if EDGE_TRIM:
        e = EDGE_TRIM
        w[:, :e, :] = w[:, -e:, :] = np.nan
        w[:, :, :e] = w[:, :, -e:] = np.nan
    return w, taux_s, tauy_s


def forcing_indicators(w, taux_s, tauy_s, mask, lat, time):
    """Indicateurs journaliers de forçage dans une région."""
    df = pd.DataFrame({
        "w_mean": region_mean(w, mask, lat),
        "w_down": region_mean(np.clip(-w, 0, None), mask, lat),
        "tau_mag": region_mean(np.hypot(taux_s, tauy_s), mask, lat),
        "tau_y": region_mean(tauy_s, mask, lat),
    }, index=time)
    ms = df.index.month.isin(SEASONS[STRONG_SEASON])
    thr = np.nanpercentile(df.loc[ms, "tau_mag"], STRONG_PCT)
    df["frac_jours_forts"] = (df["tau_mag"] > thr).astype(float).where(df["tau_mag"].notna())
    return df


def mld_series(mld, masks):
    return {r: pd.Series(region_mean(mld["mld"], m, mld["lat"]), index=mld["time"], name="MLD")
            for r, m in masks.items()}


# ======================= C. STRATIFICATION / DYNAMIQUE ======================
def stratification(masks, lon2, lat2, ds_t=None, ds_s=None, ds_uv=None):
    """Boucle temps (blocs) x tuiles. Retourne un dict de résultats (séries par région,
    cartes (période, saison), coupe à SECTION_LAT, histogrammes T-S par région/période/saison)."""
    if ds_t is None:
        ds_t, ds_s, ds_uv = open3d()
    dyn = DO_DYNAMICS and ds_uv is not None
    if dyn and "control" not in masks:
        raise ValueError("La région 'control' est requise pour les profils de référence.")
    depth_all = ds_t["depth"].values.astype(float)
    nz = int(np.searchsorted(depth_all, MAX_DEPTH, side="right"))
    depth = depth_all[:nz]
    lat = ds_t["latitude"].values.astype(float)
    lon = ds_t["longitude"].values.astype(float)
    check_grid(lon2, lat2, lon, lat, "thetao/so")
    ny, nx = len(lat), len(lon)
    time = pd.DatetimeIndex(ds_t["time"].values)
    sel = np.where((time >= pd.Timestamp(START)) & (time <= pd.Timestamp(END)))[0]
    idx = np.arange(sel[0], sel[-1] + 1, TIME_STRIDE)
    nt = len(idx)
    names = strat_names() + (dyn_names() if dyn else [])
    periods = {"P1": PERIOD1, "P2": PERIOD2}
    keys = [(p, s) for p in periods for s in MET_SEASONS]
    jrow = int(np.argmin(np.abs(lat - SECTION_LAT)))
    dz = layer_thickness(depth)
    nTS = (len(ts_axes()[0]) - 1) * (len(ts_axes()[1]) - 1)
    print(f"[strat] {nt} dates, {nz} niveaux (<= {MAX_DEPTH:.0f} m), {ny}x{nx} mailles, "
          f"tuiles {TILE}x{TILE}, blocs de {TIME_CHUNK} dates ; coupe à {lat[jrow]:.3f}N ; "
          f"dynamique : {'oui' if dyn else 'non'}")

    reg_sum = {r: {n: np.zeros(nt) for n in names} for r in masks}
    reg_cnt = {r: {n: np.zeros(nt) for n in names} for r in masks}
    map_sum = {k: {n: np.zeros((ny, nx)) for n in names} for k in keys}
    map_cnt = {k: {n: np.zeros((ny, nx)) for n in names} for k in keys}
    sec_vars = ("T", "S", "sig") + (("w", "absw") if dyn else ())
    sec = {k: {v: np.zeros((nz, nx)) for v in sec_vars + ("n",)} for k in keys}
    ts_hist = {(r,) + k: np.zeros(nTS) for r in masks for k in keys}
    mfl = {r: m.astype(float) for r, m in masks.items()}
    wgt_all = np.cos(np.deg2rad(lat))[:, None]

    for a in range(0, nt, TIME_CHUNK):
        b = min(a + TIME_CHUNK, nt)
        sl = slice(int(idx[a]), int(idx[b - 1]) + 1, TIME_STRIDE)
        tc = time[idx[a:b]]
        th = ds_t["thetao"].isel(time=sl, depth=slice(0, nz)).values      # (t, z, y, x)
        sa = ds_s["so"].isel(time=sl, depth=slice(0, nz)).values
        w = None
        if dyn:
            uu = ds_uv["uo"].isel(time=sl, depth=slice(0, nz)).values.astype("float32")
            vv = ds_uv["vo"].isel(time=sl, depth=slice(0, nz)).values.astype("float32")
            w = vertical_velocity(uu, vv, lon, lat, depth)                 # m/s, >0 vers le haut
            del uu, vv
            if EDGE_TRIM:
                e = EDGE_TRIM
                w[:, :, :e, :] = w[:, :, -e:, :] = np.nan
                w[:, :, :, :e] = w[:, :, :, -e:] = np.nan
        tsel = {k: season_mask(tc, SEASONS[k[1]], periods[k[0]]) for k in keys}

        # coupe verticale (une latitude)
        th_r = th[:, :, jrow, :].astype("float64")
        sa_r = sa[:, :, jrow, :].astype("float64")
        sg_r = compute_sigma0(sa_r[:, :, None, :], th_r[:, :, None, :], depth,
                              lat[jrow:jrow + 1], lon)[:, :, 0, :]
        w_r = (w[:, :, jrow, :].astype("float64") * 86400.0) if dyn else None
        for k, m in tsel.items():
            if m.any():
                sec[k]["T"] += np.nansum(th_r[m], 0)
                sec[k]["S"] += np.nansum(sa_r[m], 0)
                sec[k]["sig"] += np.nansum(sg_r[m], 0)
                sec[k]["n"] += np.isfinite(th_r[m]).sum(0)
                if dyn:
                    sec[k]["w"] += np.nansum(w_r[m], 0)
                    sec[k]["absw"] += np.nansum(np.abs(w_r[m]), 0)

        # boucle 1 : diagnostics de stratification + profils de référence (région control)
        store = {}
        ref_s = ref_sc = ref_w = ref_wc = 0.0
        for ys in range(0, ny, TILE):
            for xs in range(0, nx, TILE):
                ye, xe = min(ys + TILE, ny), min(xs + TILE, nx)
                th_t = th[:, :, ys:ye, xs:xe].astype("float64")
                if not np.isfinite(th_t[:, 0]).any():
                    continue                                              # tuile 100 % terre
                sa_t = sa[:, :, ys:ye, xs:xe].astype("float64")
                diags, sig = strat_diagnostics(th_t, sa_t, depth, lat[ys:ye], lon[xs:xe], True)
                store[(ys, ye, xs, xe)] = (diags, sig)
                if dyn:
                    wm = mfl["control"][ys:ye, xs:xe] * wgt_all[ys:ye]
                    if wm.any():
                        wm4 = wm[None, None]
                        fs = np.isfinite(sig)
                        ref_s = ref_s + (np.where(fs, sig, 0.0) * wm4).sum(axis=(2, 3))
                        ref_sc = ref_sc + (fs * wm4).sum(axis=(2, 3))
                        w_t = w[:, :, ys:ye, xs:xe].astype("float64")
                        fw = np.isfinite(w_t)
                        ref_w = ref_w + (np.where(fw, w_t, 0.0) * wm4).sum(axis=(2, 3))
                        ref_wc = ref_wc + (fw * wm4).sum(axis=(2, 3))
        if dyn:
            with np.errstate(invalid="ignore", divide="ignore"):
                sig_ref = np.where(ref_sc > 0, ref_s / ref_sc, np.nan)
                w_ref = np.where(ref_wc > 0, ref_w / ref_wc, np.nan)
            n2_ref = reference_n2(sig_ref, depth)

        # boucle 2 : dynamique + accumulation
        for (ys, ye, xs, xe), (diags, sig) in store.items():
            if dyn:
                diags.update(dynamics_diagnostics(
                    sig, w[:, :, ys:ye, xs:xe].astype("float64"), sig_ref, w_ref, n2_ref, depth, dz))
            th_t = th[:, :, ys:ye, xs:xe].astype("float64")
            sa_t = sa[:, :, ys:ye, xs:xe].astype("float64")
            wgt = wgt_all[ys:ye]
            for r, m in mfl.items():
                wm = m[ys:ye, xs:xe] * wgt
                if not wm.any():
                    continue
                for n, F in diags.items():
                    fin = np.isfinite(F)
                    reg_sum[r][n][a:b] += (np.where(fin, F, 0.0) * wm).sum(axis=(1, 2))
                    reg_cnt[r][n][a:b] += (fin * wm).sum(axis=(1, 2))
                mloc = masks[r][ys:ye, xs:xe]                              # diagramme T-S
                Tm, Sm = th_t[:, :, mloc], sa_t[:, :, mloc]
                for k, mt in tsel.items():
                    if mt.any():
                        add_ts(ts_hist[(r,) + k], Tm[mt], Sm[mt], dz)
            for k, mt in tsel.items():
                if mt.any():
                    for n, F in diags.items():
                        Fm = F[mt]
                        map_sum[k][n][ys:ye, xs:xe] += np.nansum(Fm, axis=0)
                        map_cnt[k][n][ys:ye, xs:xe] += np.isfinite(Fm).sum(axis=0)
        del store
        print(f"  bloc {a // TIME_CHUNK + 1}/{int(np.ceil(nt / TIME_CHUNK))}  ({tc[0].date()})")

    index = time[idx]
    series = {r: pd.DataFrame({n: np.where(reg_cnt[r][n] > 0,
                                           reg_sum[r][n] / np.maximum(reg_cnt[r][n], 1e-12),
                                           np.nan) for n in names}, index=index) for r in masks}
    maps = {k: {n: np.where(map_cnt[k][n] > 0, map_sum[k][n] / np.maximum(map_cnt[k][n], 1),
                            np.nan) for n in names} for k in keys}
    section = {}
    for k in keys:
        c = sec[k]["n"]
        section[k] = {v: np.where(c > 0, sec[k][v] / np.maximum(c, 1), np.nan) for v in sec_vars}
    return dict(series=series, maps=maps, section=section, ts=ts_hist, lon=lon, lat=lat,
                depth=depth, jrow=jrow, dynamics=dyn)


# ============================ D. STATISTIQUES ===============================
def build_annual(df, seasons):
    out = {k: pd.DataFrame({c: seasonal_mean(df[c], SEASONS[k]) for c in df.columns})
           for k in seasons}
    if "MLD" in df.columns:
        mx = df["MLD"].dropna()
        out["ANNMAX"] = pd.DataFrame({"MLD_max": mx.groupby(mx.index.year).max()})
    return out


def _periods(s):
    a = s[(s.index >= int(PERIOD1[0])) & (s.index <= int(PERIOD1[1]))].dropna()
    b = s[(s.index >= int(PERIOD2[0])) & (s.index <= int(PERIOD2[1]))].dropna()
    return a, b


def stats_table(annual, eke_ann):
    rows = []
    for season, df in annual.items():
        for c in df.columns:
            s = df[c].dropna()
            a, b = _periods(s)
            if len(a) < 5 or len(b) < 4 or len(s) < 10:
                continue
            tau, p_k = stats.kendalltau(s.index.values, s.values)
            slope = stats.theilslopes(s.values, s.index.values.astype(float))[0]
            d = pd.concat([s, eke_ann], axis=1).dropna()
            rho = rho_dt = p_dt = np.nan
            if len(d) >= 10 and d.iloc[:, 0].nunique() > 2 and d.iloc[:, 1].nunique() > 2:
                rho = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0]
                rho_dt, p_dt = stats.spearmanr(signal.detrend(d.iloc[:, 0].values),
                                               signal.detrend(d.iloc[:, 1].values))
            rows.append(dict(saison=season, variable=c, moy_P1=a.mean(), moy_P2=b.mean(),
                             var_pct=100 * (b.mean() / a.mean() - 1) if a.mean() else np.nan,
                             p_MWU=stats.mannwhitneyu(a, b).pvalue,
                             pente_pct_an=100 * slope / abs(s.mean()) if s.mean() else np.nan,
                             p_kendall=p_k, rho_EKE=rho, rho_EKE_sans_tendance=rho_dt,
                             p_sans_tendance=p_dt))
    return pd.DataFrame(rows)


# Position de la saison X de l'année y par rapport à l'EKE de SON de l'année y :
# DJF(y) = déc. y-1 à févr. y, donc avant SON(y). (saison, décalage en années, relation)
TIMINGS = (("DJF", 0, "before"), ("MAM", 0, "before"), ("JJA", 0, "before"),
           ("SON", 0, "same season"), ("DJF", 1, "after"), ("MAM", 1, "after"), ("JJA", 1, "after"))


def timing_label(s, k, rel):
    return f"{s} y{'+1' if k else ''}\n({rel})"


def leadlag_corr(annual, eke_ann, variables):
    """Corrélation de Spearman entre l'EKE de SON(y) et chaque variable saisonnière X_s(y+k)."""
    rows = []
    for v in variables:
        for s, k, rel in TIMINGS:
            df = annual.get(s)
            if df is None or v not in df:
                continue
            x = df[v].dropna().copy()
            x.index = x.index - k
            d = pd.concat([x, eke_ann], axis=1, join="inner").dropna()
            if len(d) < 10 or d.iloc[:, 0].nunique() < 4 or d.iloc[:, 1].nunique() < 4:
                continue
            rho, p = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])
            rd, pd_ = stats.spearmanr(signal.detrend(d.iloc[:, 0].values),
                                      signal.detrend(d.iloc[:, 1].values))
            rows.append(dict(variable=v, season=s, lag_years=k, timing=rel, n=len(d),
                             rho=rho, p=p, rho_detr=rd, p_detr=pd_))
    return pd.DataFrame(rows)


# ================================ LABELS (English) ==========================
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _layer_txt(n):
    return n.split("_", 1)[1][:-1] + " m"


def diag_label(n):
    """(title, axis label with unit) for a variable."""
    fixed = {"z_Smax": ("Depth of salinity maximum (LIW)", "Depth (m)"),
             "Smax": ("Maximum salinity (LIW)", "Salinity (PSS-78)"),
             "MLD": ("Mixed layer depth", "MLD (m)"),
             "MLD_max": ("Annual maximum MLD", "MLD (m)"),
             "w_mean": ("Mean Ekman vertical velocity w_E (signed)", "m/day"),
             "w_down": ("Downward Ekman pumping w_down (magnitude)", "m/day"),
             "tau_mag": ("Wind stress magnitude", "N m$^{-2}$"),
             "tau_y": ("Meridional wind stress", "N m$^{-2}$"),
             "frac_jours_forts": ("Fraction of strong-wind days", "fraction")}
    if n in fixed:
        return fixed[n]
    if n.startswith("N2_"):
        return f"N² {_layer_txt(n)}", "N² (s$^{-2}$)"
    if n.startswith("T_"):
        return f"Potential temperature {_layer_txt(n)}", "θ (°C)"
    if n.startswith("S_"):
        return f"Salinity {_layer_txt(n)}", "Salinity (PSS-78)"
    if n.startswith("z_sig"):
        return f"Depth of isopycnal σ₀ = {n[5:]}", "Depth (m)"
    if n.startswith("APE_"):
        return f"Available potential energy {_layer_txt(n)}", "APE (J m$^{-2}$)"
    if n.startswith("conv_"):
        return f"Buoyancy conversion APE→EKE {_layer_txt(n)}", "−g⟨ρ'w'⟩ (W m$^{-2}$)"
    if n.startswith("absw_"):
        return f"Mean |w| {_layer_txt(n)}", "|w| (m/day)"
    if re.match(r"w_\d", n):
        return f"Mean vertical velocity {_layer_txt(n)} (>0 up)", "w (m/day)"
    return n, n


def season_xlabel(season):
    return "Year (winter: year of January)" if season == "DJF" else "Year"


def season_title(season):
    return f"{season} ({'-'.join(MONTH_ABBR[m - 1] for m in SEASONS[season])})"


def key_diags():
    z, c = LAYERS[1], LAYERS[2]
    return [f"z_sig{ISOPYCNALS[1]}", f"N2_{z[0]}-{z[1]}m", f"T_{z[0]}-{z[1]}m",
            "Smax", "z_Smax", f"N2_{c[0]}-{c[1]}m"]


def key_dyn():
    z, c = LAYERS[1], LAYERS[2]
    return [f"APE_{z[0]}-{z[1]}m", f"APE_{c[0]}-{c[1]}m", f"absw_{z[0]}-{z[1]}m",
            f"w_{z[0]}-{z[1]}m", f"conv_{z[0]}-{z[1]}m", f"conv_{c[0]}-{c[1]}m"]


def _eddy_label():
    return f"Eddy region (EKE ≥ {EKE_ISOLINE:.0f} cm² s⁻²)"


def _control_label():
    return f"Control region (EKE < {CONTROL_EKE:.0f} cm² s⁻²)"


def _lobe_label():
    return f"Pumping lobe (w_E ≤ {LOBE_THRESHOLD} m/day, {LOBE_SEASON})"


# ================================ FIGURES ===================================
def _outline(ax, lon, lat, mask, color, ls="-"):
    ax.contour(lon, lat, mask.astype(float), levels=[0.5], colors=color,
               linestyles=ls, linewidths=1.5)


def _mld_line(ax, lon, mld):
    """MLD : thin white dashed line."""
    ax.plot(lon, mld, color="w", lw=1.3, ls=(0, (4, 3)))


def plot_regions(eke_map, w_clim, w_diff, masks, lon, lat, path):
    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    axs = axs.ravel()
    im = axs[0].pcolormesh(lon, lat, eke_map, cmap="viridis", shading="auto")
    plt.colorbar(im, ax=axs[0], label="Mean EKE (cm$^2$ s$^{-2}$)")
    axs[0].contour(lon, lat, eke_map, levels=[EKE_ISOLINE], colors="w", linewidths=1)
    _outline(axs[0], lon, lat, masks["eddy"], "r")
    _outline(axs[0], lon, lat, masks["control"], "k", "--")
    _outline(axs[0], lon, lat, masks["lobe"], "m")
    axs[0].legend(handles=[Line2D([], [], color="r", label=_eddy_label()),
                           Line2D([], [], color="k", ls="--", label=_control_label()),
                           Line2D([], [], color="m", label=_lobe_label())],
                  loc="lower left", fontsize=7)
    axs[0].set_title("Mean EKE and analysis regions")
    v = np.nanpercentile(np.abs(np.concatenate([c.ravel() for c in w_clim.values()])), 98)
    for ax, s in zip(axs[1:5], MET_SEASONS):
        im = ax.pcolormesh(lon, lat, w_clim[s], cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
        plt.colorbar(im, ax=ax, label="Mean w_E (m/day; >0 suction, <0 pumping)")
        _outline(ax, lon, lat, masks["eddy"], "k")
        _outline(ax, lon, lat, masks["lobe"], "m")
        ax.set_title(f"Ekman vertical velocity w_E, {season_title(s)}")
    dv = np.nanpercentile(np.abs(w_diff), 98)
    im = axs[5].pcolormesh(lon, lat, w_diff, cmap="RdBu_r", vmin=-dv, vmax=dv, shading="auto")
    plt.colorbar(im, ax=axs[5], label="Δ w_E (m/day)  (P2 − P1)")
    _outline(axs[5], lon, lat, masks["eddy"], "k")
    _outline(axs[5], lon, lat, masks["lobe"], "m")
    axs[5].set_title(f"w_E {LOBE_SEASON}: {PERIOD2[0]}-{PERIOD2[1]} minus {PERIOD1[0]}-{PERIOD1[1]}")
    for ax in axs:
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_forcing(annual_lobe, eke_ann, path):
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    axs = axs.ravel()
    handles = None
    for ax, s in zip(axs, MET_SEASONS):
        df = annual_lobe[s]
        h1, = ax.plot(df.index, df["w_mean"], "o-", ms=3, color="tab:blue",
                      label="Mean w_E (signed; <0 = downward)")
        h2, = ax.plot(df.index, df["w_down"], "o-", ms=3, color="tab:cyan",
                      label="w_down: downward pumping magnitude (>0 = downward)")
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(2012.5, color="grey", ls=":")
        ax.set_xlabel(season_xlabel(s))
        ax.set_ylabel("w (m/day)")
        ax.set_title(f"Forcing in the pumping lobe, {season_title(s)}, vs EKE {EKE_SEASON}")
        ax2 = ax.twinx()
        h3, = ax2.plot(eke_ann.index, eke_ann.values, "s-", ms=3, color="tab:red", alpha=0.7,
                       label=f"EKE {EKE_SEASON} (eddy region)")
        ax2.set_ylabel("EKE (cm$^2$ s$^{-2}$)", color="tab:red")
        handles = [h1, h2, h3]
    fig.legend(handles=handles + [Line2D([], [], color="grey", ls=":", label="2012/2013 (P1/P2 split)")],
               loc="lower center", ncol=2, fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_series_grid(annual, eke_ann, season, names, suptitle, path):
    fig, axs = plt.subplots(2, 4, figsize=(19, 8))
    axs = axs.ravel()
    axs[0].plot(eke_ann.index, eke_ann.values, "s-", ms=3, color="tab:red", label="Eddy region")
    axs[0].set_title(f"EKE {EKE_SEASON}")
    axs[0].set_ylabel("EKE (cm$^2$ s$^{-2}$)")
    axs[0].set_xlabel("Year")
    axs[0].legend(fontsize=8)
    for ax, n in zip(axs[1:], names):
        for r, ls, col, lab in (("eddy", "-", "tab:blue", _eddy_label()),
                                ("control", "--", "tab:grey", _control_label())):
            df = annual[r].get(season)
            if df is not None and n in df:
                ax.plot(df.index, df[n], ls, marker="o", ms=3, color=col, label=lab)
        title, ylab = diag_label(n)
        ax.set_title(f"{title}\n{season_title(season)}", fontsize=9)
        ax.set_ylabel(ylab)
        ax.set_xlabel(season_xlabel(season))
        ax.axvline(2012.5, color="grey", ls=":")
        if n.startswith("z_"):
            ax.invert_yaxis()
        ax.legend(fontsize=7)
    for ax in axs[len(names) + 1:]:
        ax.axis("off")
    fig.suptitle(f"{suptitle}, {season_title(season)}: eddy vs control region "
                 "(grey dotted line: 2012/2013 split)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_maps_diff(maps, masks, lon, lat, season, names, path):
    fig, axs = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4.8), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, n in zip(axs, names):
        d = maps[("P2", season)][n] - maps[("P1", season)][n]
        v = np.nanpercentile(np.abs(d), 98) if np.isfinite(d).any() else 1.0
        im = ax.pcolormesh(lon, lat, d, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
        _outline(ax, lon, lat, masks["eddy"], "k")
        title, ylab = diag_label(n)
        plt.colorbar(im, ax=ax, label=f"Δ {ylab}  (P2 − P1)")
        ax.set_title(f"{title}\n{season_title(season)}: {PERIOD2[0]}-{PERIOD2[1]} minus "
                     f"{PERIOD1[0]}-{PERIOD1[1]}", fontsize=8)
        ax.set_xlabel("Longitude (°E)")
    axs[0].set_ylabel("Latitude (°N)")
    axs[0].legend(handles=[Line2D([], [], color="k", label=_eddy_label())],
                  loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_section(res, mld_sec, masks, season, path):
    """Zonal section at SECTION_LAT: T (top) and S (bottom) as shading, isopycnals, MLD."""
    depth, lon, j = res["depth"], res["lon"], res["jrow"]
    kz = depth <= SECTION_ZMAX
    z = depth[kz]
    s1, s2 = res["section"][("P1", season)], res["section"][("P2", season)]
    row_eddy = masks["eddy"][j, :]
    fig, axs = plt.subplots(2, 3, figsize=(20, 9.5), sharex=True, sharey=True)
    for r, (var, cmap, cb, dcb) in enumerate((
            ("T", "RdYlBu_r", "Potential temperature (°C)", "Δθ (°C)  (P2 − P1)"),
            ("S", "viridis", "Salinity (PSS-78)", "ΔSalinity (PSS-78)  (P2 − P1)"))):
        lo = np.nanmin([np.nanmin(s1[var][kz]), np.nanmin(s2[var][kz])])
        hi = np.nanmax([np.nanmax(s1[var][kz]), np.nanmax(s2[var][kz])])
        for c, (sec, per, p) in enumerate(((s1, PERIOD1, "P1"), (s2, PERIOD2, "P2"))):
            ax = axs[r, c]
            im = ax.pcolormesh(lon, z, sec[var][kz], cmap=cmap, vmin=lo, vmax=hi, shading="auto")
            cs = ax.contour(lon, z, sec["sig"][kz], levels=SECTION_ISOPYCNALS, colors="k", linewidths=0.6)
            ax.clabel(cs, fmt="%.2f", fontsize=SECTION_LABEL_FS, inline_spacing=6)
            ax.contour(lon, z, sec["sig"][kz], levels=ISOPYCNALS, colors="k", linewidths=1.8)
            _mld_line(ax, lon, mld_sec[(p, season)])
            ax.fill_between(lon, 0, 1, where=row_eddy, transform=ax.get_xaxis_transform(),
                            color="tab:red", alpha=0.12, lw=0)
            ax.set_title(f"{p} ({per[0]}-{per[1]}), {season_title(season)}", fontsize=11)
            plt.colorbar(im, ax=ax, label=cb)
        ax = axs[r, 2]
        d = s2[var][kz] - s1[var][kz]
        v = np.nanpercentile(np.abs(d), 98) if np.isfinite(d).any() else 1.0
        im = ax.pcolormesh(lon, z, d, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
        ax.contour(lon, z, s1["sig"][kz], levels=ISOPYCNALS, colors="k", linewidths=1.2, linestyles="--")
        ax.contour(lon, z, s2["sig"][kz], levels=ISOPYCNALS, colors="k", linewidths=1.2)
        ax.plot(lon, mld_sec[("P1", season)], color="k", lw=1.5, ls="--")
        ax.plot(lon, mld_sec[("P2", season)], color="k", lw=1.5)
        ax.fill_between(lon, 0, 1, where=row_eddy, transform=ax.get_xaxis_transform(),
                        color="tab:red", alpha=0.12, lw=0)
        ax.set_title(f"Difference P2 − P1 ({'temperature' if var == 'T' else 'salinity'})", fontsize=11)
        plt.colorbar(im, ax=ax, label=dcb)
    for ax in axs[1, :]:
        ax.set_xlabel("Longitude (°E)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Depth (m)")
    axs[0, 0].set_ylim(SECTION_ZMAX, 0)
    handles = [Line2D([], [], color="k", lw=0.6, label="Isopycnals σ₀ (0.25 kg m$^{-3}$ step)"),
               Line2D([], [], color="k", lw=1.8, label="σ₀ = " + ", ".join(str(s) for s in ISOPYCNALS)),
               Line2D([], [], color="0.55", lw=1.3, ls=(0, (4, 3)), label="Mean MLD (white dashed line)"),
               Line2D([], [], color="k", ls="--", lw=1.2, label="Right column: P1 dashed / P2 solid"),
               Patch(color="tab:red", alpha=0.25, label=_eddy_label())]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10, frameon=True)
    fig.suptitle(f"Zonal section at {res['lat'][j]:.2f}°N: temperature and salinity (shading), "
                 f"isopycnals and MLD, {season_title(season)}")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_section_w(res, masks, season, path):
    """Zonal section at SECTION_LAT of the diagnosed vertical velocity (mean and mean |w|)."""
    depth, lon, j = res["depth"], res["lon"], res["jrow"]
    kz = depth <= SECTION_ZMAX
    z = depth[kz]
    s1, s2 = res["section"][("P1", season)], res["section"][("P2", season)]
    row_eddy = masks["eddy"][j, :]
    fig, axs = plt.subplots(2, 3, figsize=(20, 9), sharex=True, sharey=True)
    for r, (var, cmap, cb, dcb, sym) in enumerate((
            ("w", "RdBu_r", "Mean w (m/day; >0 upward)", "Δ mean w (m/day)  (P2 − P1)", True),
            ("absw", "magma_r", "Mean |w| (m/day)", "Δ mean |w| (m/day)  (P2 − P1)", False))):
        both = np.concatenate([s1[var][kz].ravel(), s2[var][kz].ravel()])
        v = np.nanpercentile(np.abs(both), 98) if np.isfinite(both).any() else 1.0
        for c, (sec, per, p) in enumerate(((s1, PERIOD1, "P1"), (s2, PERIOD2, "P2"))):
            ax = axs[r, c]
            im = ax.pcolormesh(lon, z, sec[var][kz], cmap=cmap, vmin=-v if sym else 0, vmax=v, shading="auto")
            ax.contour(lon, z, sec["sig"][kz], levels=ISOPYCNALS, colors="k", linewidths=1.5)
            ax.fill_between(lon, 0, 1, where=row_eddy, transform=ax.get_xaxis_transform(),
                            color="tab:red", alpha=0.12, lw=0)
            ax.set_title(f"{p} ({per[0]}-{per[1]}), {season_title(season)}", fontsize=11)
            plt.colorbar(im, ax=ax, label=cb)
        ax = axs[r, 2]
        d = s2[var][kz] - s1[var][kz]
        dv = np.nanpercentile(np.abs(d), 98) if np.isfinite(d).any() else 1.0
        im = ax.pcolormesh(lon, z, d, cmap="RdBu_r", vmin=-dv, vmax=dv, shading="auto")
        ax.contour(lon, z, s1["sig"][kz], levels=ISOPYCNALS, colors="k", linewidths=1.2, linestyles="--")
        ax.contour(lon, z, s2["sig"][kz], levels=ISOPYCNALS, colors="k", linewidths=1.2)
        ax.fill_between(lon, 0, 1, where=row_eddy, transform=ax.get_xaxis_transform(),
                        color="tab:red", alpha=0.12, lw=0)
        ax.set_title("Difference P2 − P1", fontsize=11)
        plt.colorbar(im, ax=ax, label=dcb)
    for ax in axs[1, :]:
        ax.set_xlabel("Longitude (°E)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Depth (m)")
    axs[0, 0].set_ylim(SECTION_ZMAX, 0)
    handles = [Line2D([], [], color="k", lw=1.5, label="σ₀ = " + ", ".join(str(s) for s in ISOPYCNALS)),
               Line2D([], [], color="k", ls="--", lw=1.2, label="Right column: P1 dashed / P2 solid"),
               Patch(color="tab:red", alpha=0.25, label=_eddy_label())]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, frameon=True)
    fig.suptitle(f"Vertical velocity from continuity, zonal section at {res['lat'][j]:.2f}°N, "
                 f"{season_title(season)} (seasonal means of daily fields)")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _hpd_level(h, frac):
    """Density threshold such that {h >= threshold} contains the fraction frac of the volume."""
    f = np.sort(h.ravel())[::-1]
    c = np.cumsum(f) / f.sum()
    return f[min(np.searchsorted(c, frac), len(f) - 1)]


def plot_ts(res, season, path):
    tg, sg = ts_axes()
    nT, nS = len(tg) - 1, len(sg) - 1
    Tc, Sc = 0.5 * (tg[1:] + tg[:-1]), 0.5 * (sg[1:] + sg[:-1])
    TT, SS = np.meshgrid(Tc, Sc, indexing="ij")
    sig = compute_sigma0(SS[None, None], TT[None, None], np.array([0.0]),
                         np.full(nT, float(np.mean(res["lat"]))),
                         np.full(nS, float(np.mean(res["lon"]))))[0, 0]

    def H(r, periods=("P1", "P2")):
        h = sum(res["ts"][(r, p, season)] for p in periods).reshape(nT, nS)
        return h / h.sum() if h.sum() > 0 else h

    def smooth(h):
        return ndimage.gaussian_filter(h, 1.0)

    def contours(ax, h, color):
        hs = smooth(h)
        if hs.sum() <= 0:
            return
        lv = sorted({_hpd_level(hs, 0.9), _hpd_level(hs, 0.5)})
        ax.contour(Sc, Tc, hs, levels=lv, colors=color, linewidths=1.6)

    he, hc = H("eddy"), H("control")
    fig, axs = plt.subplots(1, 3, figsize=(20, 6.6), sharex=True, sharey=True)
    ax = axs[0]
    with np.errstate(divide="ignore"):
        dens = np.where(he > 0, np.log10(he), np.nan)
    im = ax.pcolormesh(Sc, Tc, dens, cmap="viridis", vmin=-5,
                       vmax=np.nanmax(dens) if np.isfinite(dens).any() else 0, shading="auto")
    plt.colorbar(im, ax=ax, label="log$_{10}$ (volume fraction), eddy region")
    contours(ax, hc, "tab:red")
    ax.legend(handles=[Line2D([], [], color="tab:red", label="Control region: contours enclosing 50% and 90% of its volume")],
              loc="upper left", fontsize=8)
    ax.set_title("Eddy region (colour) vs control region (red contours)", fontsize=10)
    ax = axs[1]
    d = smooth(he) - smooth(hc)
    v = np.nanpercentile(np.abs(d[d != 0]), 98) if np.any(d != 0) else 1.0
    im = ax.pcolormesh(Sc, Tc, d, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
    plt.colorbar(im, ax=ax, label="Δ volume fraction (eddy − control)")
    ax.set_title("Eddy minus control (red: more frequent in the eddy region)", fontsize=10)
    ax = axs[2]
    contours(ax, H("eddy", ("P1",)), "tab:blue")
    contours(ax, H("eddy", ("P2",)), "tab:orange")
    ax.legend(handles=[Line2D([], [], color="tab:blue", label=f"Eddy region, P1 ({PERIOD1[0]}-{PERIOD1[1]})"),
                       Line2D([], [], color="tab:orange", label=f"Eddy region, P2 ({PERIOD2[0]}-{PERIOD2[1]})")],
              loc="upper left", fontsize=9, title="Contours: 50% and 90% of the volume")
    ax.set_title("Eddy region: P1 vs P2", fontsize=10)
    for ax in axs:
        cs = ax.contour(Sc, Tc, sig, levels=np.arange(22.0, 30.0, 0.5), colors="grey",
                        linewidths=0.6, linestyles=":")
        ax.clabel(cs, fmt="%.1f", fontsize=8)
        ax.set_xlabel("Salinity (PSS-78)")
    axs[0].set_ylabel("Potential temperature (°C)")
    occ = (he + hc) > 0
    if occ.any():
        ti, si = np.where(occ)
        axs[0].set_xlim(Sc[si.min()] - 0.1, Sc[si.max()] + 0.1)
        axs[0].set_ylim(Tc[ti.min()] - 0.5, Tc[ti.max()] + 0.5)
    fig.suptitle(f"θ-S diagrams, {season_title(season)}, 0-{MAX_DEPTH:.0f} m. Each bin = share of the region's "
                 "water volume (weighted by layer thickness). Dotted grey: isopycnals σ₀ (kg m$^{-3}$)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_leadlag(ll, variables, title, path):
    """Heatmap of the detrended Spearman correlation between EKE(SON, y) and X_season(y+k)."""
    regions = [(r, lab) for r, lab in (("control", _control_label()), ("eddy", _eddy_label()))
               if r in ll and len(ll[r])]
    if not regions:
        return
    fig, axs = plt.subplots(1, len(regions), figsize=(9.5 * len(regions), 0.5 * len(variables) + 4.2),
                            sharey=True, squeeze=False)
    for ax, (r, lab) in zip(axs[0], regions):
        tab = ll[r]
        mat = np.full((len(variables), len(TIMINGS)), np.nan)
        pm = np.full_like(mat, np.nan)
        for i, v in enumerate(variables):
            for j, (s, k, rel) in enumerate(TIMINGS):
                row = tab[(tab["variable"] == v) & (tab["season"] == s) & (tab["lag_years"] == k)]
                if len(row):
                    mat[i, j], pm[i, j] = row["rho_detr"].iloc[0], row["p_detr"].iloc[0]
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}" + ("*" if pm[i, j] < 0.05 else ""),
                            ha="center", va="center", fontsize=8)
        ax.axvline(2.5, color="k", lw=2)
        ax.axvline(3.5, color="k", lw=2)
        ax.set_xticks(range(len(TIMINGS)))
        ax.set_xticklabels([timing_label(*t) for t in TIMINGS], fontsize=8)
        ax.set_yticks(range(len(variables)))
        ax.set_yticklabels([diag_label(v)[0] for v in variables], fontsize=8)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel(f"Season of the predictor, relative to EKE {EKE_SEASON} of year y")
        plt.colorbar(im, ax=ax, label="Spearman ρ (detrended series)")
    fig.suptitle(f"{title}\nCorrelation with EKE {EKE_SEASON} (eddy region); "
                 "* : p < 0.05 (not corrected for multiple tests)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ================================== MAIN ====================================
def main():
    print(f"[version] {VERSION}  |  fichier : {os.path.abspath(__file__)}")
    os.makedirs(OUTDIR, exist_ok=True)
    # --- A. EKE et régions eddy / control
    vel = load_vg()
    lon, lat = vel["lon"], vel["lat"]
    eke_t, eke_map = compute_eke(vel)
    masks = build_masks(eke_map)
    eke_daily = pd.Series(region_mean(eke_t, masks["eddy"], lat), index=vel["time"])
    eke_ann = seasonal_mean(eke_daily, SEASONS[EKE_SEASON])
    eke_ann.to_csv(os.path.join(OUTDIR, f"eke_{EKE_SEASON}_region_eddy.csv"))
    del eke_t, vel

    # --- B. Pompage d'Ekman, lobe, forçage
    mflux = load_mflux()
    check_grid(lon, lat, mflux["lon"], mflux["lat"], "mflux")
    w, taux_s, tauy_s = ekman_fields(mflux)
    t = mflux["time"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        w_clim = {s: np.nanmean(w[np.asarray(t.month.isin(SEASONS[s]))], axis=0) for s in MET_SEASONS}
        mj = season_mask(t, SEASONS[LOBE_SEASON], (PERIOD1[0], PERIOD2[1]))
        w_diff = (np.nanmean(w[season_mask(t, SEASONS[LOBE_SEASON], PERIOD2)], axis=0)
                  - np.nanmean(w[season_mask(t, SEASONS[LOBE_SEASON], PERIOD1)], axis=0))
        w_lobe_clim = np.nanmean(w[mj], axis=0)
    masks["lobe"] = build_lobe(w_lobe_clim, lon, lat)
    print(f"[lobe] {masks['lobe'].sum()} mailles (w_E <= {LOBE_THRESHOLD} m/jour en {LOBE_SEASON})")
    forcing = {r: forcing_indicators(w, taux_s, tauy_s, masks[r], lat, t) for r in ("eddy", "lobe")}
    np.savez(os.path.join(OUTDIR, "masques.npz"), eke_map=eke_map, lon=lon, lat=lat, **masks)
    plot_regions(eke_map, w_clim, w_diff, masks, lon, lat, os.path.join(OUTDIR, "regions_et_w_Ekman.png"))
    del w, taux_s, tauy_s
    mld = load_mld()
    check_grid(lon, lat, mld["lon"], mld["lat"], "mld")
    mld_ts = mld_series(mld, masks)

    # --- C. Stratification et dynamique
    pkl = os.path.join(OUTDIR, "strat_results.pkl")
    if RECOMPUTE_STRAT or not os.path.exists(pkl):
        res = stratification(masks, lon, lat)
        pd.to_pickle(res, pkl)
    else:
        res = pd.read_pickle(pkl)
    dyn = res.get("dynamics", False)
    j = res["jrow"]
    mld_sec = {}
    for p, per in (("P1", PERIOD1), ("P2", PERIOD2)):
        for s in MET_SEASONS:
            m = season_mask(mld["time"], SEASONS[s], per)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mld_sec[(p, s)] = np.nanmean(mld["mld"][m][:, j, :], axis=0)

    # --- D. Statistiques
    dfs = {"eddy": pd.concat([forcing["eddy"], mld_ts["eddy"], res["series"]["eddy"]], axis=1),
           "lobe": pd.concat([forcing["lobe"], mld_ts["lobe"], res["series"]["lobe"]], axis=1),
           "control": pd.concat([mld_ts["control"], res["series"]["control"]], axis=1)}
    annual = {r: build_annual(df, MET_SEASONS) for r, df in dfs.items()}
    pd.set_option("display.width", 220)
    show = set(key_diags()) | set(key_dyn() if dyn else []) | \
        {"w_mean", "w_down", "tau_mag", "frac_jours_forts", "MLD", "MLD_max"}
    for r in annual:
        tab = stats_table(annual[r], eke_ann)
        tab.to_csv(os.path.join(OUTDIR, f"stats_{r}.csv"), index=False)
        print(f"\n===== Région {r} : P1 {PERIOD1} vs P2 {PERIOD2} (variables clés ; tout dans stats_{r}.csv) =====")
        print(tab[tab["variable"].isin(show)].to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # corrélations décalées avec l'EKE de SON
    strat_vars = [n for n in strat_names() if n.startswith(("N2_", "z_sig"))]
    dyn_vars = dyn_names() if dyn else []
    ll = {r: leadlag_corr(annual[r], eke_ann, strat_vars + dyn_vars) for r in ("control", "eddy")}
    for r, tab in ll.items():
        tab.to_csv(os.path.join(OUTDIR, f"leadlag_EKE_{EKE_SEASON}_{r}.csv"), index=False)
    if len(ll["control"]):
        top = ll["control"][ll["control"]["variable"].isin(strat_vars)]
        top = top.reindex(top["rho_detr"].abs().sort_values(ascending=False).index).head(8)
        print(f"\n--- Région témoin : 8 plus fortes corrélations (sans tendance) de N2/isopycnes avec l'EKE {EKE_SEASON} ---")
        print(f"(attention : ~{len(strat_vars) * len(TIMINGS)} tests, des corrélations fortes apparaissent par hasard)")
        print(top[["variable", "season", "lag_years", "timing", "n", "rho_detr", "p_detr"]]
              .to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    # --- Figures
    plot_forcing(annual["lobe"], eke_ann, os.path.join(OUTDIR, "forcage_lobe_vs_EKE.png"))
    plot_leadlag(ll, strat_vars, "Stratification as predictor (N², isopycnal depth)",
                 os.path.join(OUTDIR, f"corr_EKE_{EKE_SEASON}_stratification.png"))
    if dyn:
        plot_leadlag(ll, dyn_vars, "APE, vertical velocity and buoyancy conversion",
                     os.path.join(OUTDIR, f"corr_EKE_{EKE_SEASON}_dynamics.png"))
    for s in MET_SEASONS:
        plot_series_grid(annual, eke_ann, s, key_diags(), "Stratification",
                         os.path.join(OUTDIR, f"stratification_series_{s}.png"))
        plot_maps_diff(res["maps"], masks, res["lon"], res["lat"], s, key_diags()[:4],
                       os.path.join(OUTDIR, f"stratification_cartes_{s}.png"))
        plot_section(res, mld_sec, masks, s, os.path.join(OUTDIR, f"coupe_{SECTION_LAT}N_{s}.png"))
        plot_ts(res, s, os.path.join(OUTDIR, f"diagramme_TS_{s}.png"))
        if dyn:
            kd = key_dyn()
            plot_series_grid(annual, eke_ann, s, kd, "APE, vertical velocity, buoyancy conversion",
                             os.path.join(OUTDIR, f"dynamique_series_{s}.png"))
            plot_maps_diff(res["maps"], masks, res["lon"], res["lat"], s,
                           [kd[0], kd[2], kd[3], kd[4]],
                           os.path.join(OUTDIR, f"dynamique_cartes_{s}.png"))
            plot_section_w(res, masks, s, os.path.join(OUTDIR, f"coupe_w_{SECTION_LAT}N_{s}.png"))
    print(f"\nSorties dans {OUTDIR}/")


if __name__ == "__main__":
    main()
