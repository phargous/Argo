#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ierapetra (modèle MedMFC seul) : forçage par le vent + stratification, toutes saisons.

  A. EKE du modèle (ug, vg géostrophiques) -> carte moyenne -> région "eddy" =
     composante connexe de l'isoligne EKE >= EKE_ISOLINE ; région "control" = EKE moyenne faible.
  B. Forçage dans la région eddy : w_E, w_down, module du stress, fraction de jours forts.
  C. Stratification (so, thetao, 0-MAX_DEPTH m) sans jamais charger les fichiers en entier :
     blocs de temps x tuiles de 10x10 mailles ; chaque diagnostic est réduit aussitôt en
     séries par région, cartes, coupe verticale (SECTION_LAT) et histogrammes T-S pondérés
     par l'épaisseur des couches, par période et par saison.
  D. Statistiques (tendances, P1 vs P2, corrélations avec l'EKE de SON) et figures.

Saisons : DJF (déc. de l'année précédente + janv. + févr., étiquetée par l'année de janvier),
MAM, JJA, SON ; plus JAS (juil.-sept.), saison de forçage qui précède l'EKE de SON.

  w_E     = (1/rho0) * [ d(tau_y/f)/dx - d(tau_x/f)/dy ]   (m/jour, >0 succion, <0 pompage)
  w_down  = moyenne spatiale de max(-w_E, 0) : intensité du seul pompage descendant
  N2      = (g/rho0) * d(sigma0)/dz     (z positif vers le bas)

Dépendances : numpy, pandas, scipy, matplotlib, xarray, gsw.
Lancer :  python ierapetra_forcage_stratification.py
"""
import os
import warnings
import numpy as np
import pandas as pd
from scipy import ndimage, stats, signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as pe

# =============================== CONFIG ====================================
VG_FILE = "../../results/006_004_medmfc_vg_iera.nc"
UG_VAR, VG_VAR = "ug", "vg"
VG_FACTOR = 1.0                     # 0.01 si les vitesses sont en cm/s
DATA = "../../data/"
MFLUX_FILE = DATA + "006_004_medmfc_mflux_iera.nc"
MLD_FILE = DATA + "006_004_medmfc_mld_iera.nc"
THETAO_FILE = DATA + "006_004_medmfc_thetao_iera.nc"
SO_FILE = DATA + "006_004_medmfc_so_iera.nc"

MODEL_TAUX, MODEL_TAUY = None, None  # None = auto (standard_name / noms usuels)
MODEL_SIGN = 1.0                    # -1 si le contrôle de signe échoue
MLD_VAR = None                      # None = auto (mlotst, mld, sinon 1re variable)

START, END = "1993-01-01", "2023-12-31"   # les fichiers vont jusqu'en 2026
PERIOD1 = ("1993", "2012")
PERIOD2 = ("2013", "2023")

# --- région par isoligne d'EKE ---
EKE_ISOLINE = 300.0                 # cm2/s2, sur l'EKE moyenne de BOX_PERIOD
BOX_PERIOD = (START, END)
EKE_REF_PERIOD = (START, END)       # référence des anomalies de vitesse
CONTROL_EKE = 150.0                 # région témoin : EKE moyenne < CONTROL_EKE

# --- saisons ---
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11),
           "JAS": (7, 8, 9)}
MET_SEASONS = ("DJF", "MAM", "JJA", "SON")     # saisons des cartes, coupes, diagrammes T-S
FORCING_SEASONS = ("DJF", "MAM", "JJA", "JAS", "SON")
EKE_SEASON = "SON"                  # EKE de référence (corrélations)
STRONG_SEASON = "JAS"               # saison fixant le seuil des "jours forts"

# --- forçage ---
SMOOTH_DEG_MODEL = 0.25
EDGE_TRIM = 2
STRONG_PCT = 90

# --- stratification ---
MAX_DEPTH = 1000.0                  # m
TILE = 10                           # tuiles de 10x10 mailles
TIME_CHUNK = 30                     # dates lues à la fois
TIME_STRIDE = 5                     # 1 = toutes les dates (long) ; 5 = une sur 5 (rapide)
LAYERS = ((0, 100), (100, 300), (300, 700), (700, 1000))
ISOPYCNALS = (27.5, 28.5, 29.0)     # sigma0 suivies (séries, coupes)
LIW_RANGE = (150.0, 700.0)
RECOMPUTE_STRAT = True              # False : relire out/strat_results.pkl

# --- coupe verticale et T-S ---
SECTION_LAT = 34.25
SECTION_ZMAX = 700.0                # m, profondeur max affichée sur la coupe
SECTION_ISOPYCNALS = np.arange(26.0, 29.5, 0.25)
SECTION_LABEL_FS = 11               # taille du texte des labels d'isopycnes (coupes)
TS_T = (12.0, 30.0, 0.1)            # (min, max, pas) température potentielle
TS_S = (38.2, 39.8, 0.01)           # (min, max, pas) salinité

OUTDIR = "out_ierapetra"
RHO0, G = 1025.0, 9.81
OMEGA, R_EARTH = 7.2921e-5, 6.371e6


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
    """(t, y, x) en N m-2 -> w_E en m/s."""
    phi = np.deg2rad(lat)
    f = (2.0 * OMEGA * np.sin(phi))[None, :, None]
    dty_dx = np.gradient(tauy / f, np.deg2rad(lon), axis=2) \
        / (R_EARTH * np.cos(phi))[None, :, None]
    dtx_dy = np.gradient(taux / f, phi, axis=1) / R_EARTH
    return (dty_dx - dtx_dy) / RHO0


def region_mean(a, mask, lat):
    """Moyenne pondérée cos(lat) de a (t,y,x) sur mask (y,x), tolérante aux NaN."""
    wm = mask.astype(float) * np.cos(np.deg2rad(lat))[:, None]
    fin = np.isfinite(a)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.where(fin, a, 0.0) * wm).sum(axis=(1, 2)) / (fin * wm).sum(axis=(1, 2))


def largest_region_at_max(eke_map, threshold):
    """Composante connexe de {EKE >= threshold} contenant le maximum d'EKE."""
    cond = np.isfinite(eke_map) & (eke_map >= threshold)
    if not cond.any():
        raise ValueError(f"Aucune maille avec EKE >= {threshold} (max = "
                         f"{np.nanmax(eke_map):.0f} cm2/s2). Baisser EKE_ISOLINE.")
    lab, _ = ndimage.label(cond, structure=np.ones((3, 3)))
    imax = np.unravel_index(np.nanargmax(np.where(cond, eke_map, -np.inf)), eke_map.shape)
    return lab == lab[imax]


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


def diag_names():
    names = []
    for z1, z2 in LAYERS:
        names += [f"N2_{z1}-{z2}m", f"T_{z1}-{z2}m", f"S_{z1}-{z2}m"]
    names += [f"z_sig{s}" for s in ISOPYCNALS]
    return names + ["z_Smax", "Smax"]


def strat_diagnostics(theta, sal, depth, lat, lon):
    """theta, sal : (t, z, y, x) float64 -> dict nom -> (t, y, x).
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


def open3d():
    """Ouvre (sans rien charger) thetao et so ; vérifie la cohérence des axes."""
    import xarray as xr
    dt = _std_coords(xr.open_dataset(THETAO_FILE))
    ds = _std_coords(xr.open_dataset(SO_FILE))
    for c in ("time", "depth", "latitude", "longitude"):
        if not np.array_equal(dt[c].values, ds[c].values):
            raise ValueError(f"Axe '{c}' différent entre thetao et so.")
    for c in ("latitude", "longitude", "depth"):
        if not np.all(np.diff(dt[c].values) > 0):
            raise ValueError(f"'{c}' doit être croissant (trier les fichiers).")
    return dt, ds


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
def forcing_series(mflux, masks):
    """w_E (t,y,x en m/jour) et indicateurs journaliers dans la région eddy."""
    time, lon, lat = mflux["time"], mflux["lon"], mflux["lat"]
    taux, tauy = mflux["taux"], mflux["tauy"]
    m = np.asarray(time.month.isin(SEASONS["JJA"]))
    print(f"[contrôle] tau_y moyen JJA : {np.nanmean(tauy[m]):+.4f} N/m2 "
          "(attendu < 0 : Étésiens de nord). Sinon MODEL_SIGN = -1.")
    dlon, dlat = np.abs(np.diff(lon)).mean(), np.abs(np.diff(lat)).mean()
    sy, sx = SMOOTH_DEG_MODEL / dlat, SMOOTH_DEG_MODEL / dlon
    taux_s, tauy_s = nan_gaussian(taux, sy, sx), nan_gaussian(tauy, sy, sx)
    w = ekman_pumping_np(taux_s, tauy_s, lon, lat) * 86400.0
    if EDGE_TRIM:
        e = EDGE_TRIM
        w[:, :e, :] = w[:, -e:, :] = np.nan
        w[:, :, :e] = w[:, :, -e:] = np.nan
    eddy = masks["eddy"]
    df = pd.DataFrame({
        "w_mean": region_mean(w, eddy, lat),
        "w_down": region_mean(np.clip(-w, 0, None), eddy, lat),
        "tau_mag": region_mean(np.hypot(taux_s, tauy_s), eddy, lat),
        "tau_y": region_mean(tauy_s, eddy, lat),
    }, index=time)
    ms = df.index.month.isin(SEASONS[STRONG_SEASON])
    thr = np.nanpercentile(df.loc[ms, "tau_mag"], STRONG_PCT)
    df["frac_jours_forts"] = (df["tau_mag"] > thr).astype(float).where(df["tau_mag"].notna())
    return df, w


def mld_series(mld, masks):
    return {r: pd.Series(region_mean(mld["mld"], m, mld["lat"]), index=mld["time"], name="MLD")
            for r, m in masks.items()}


# ======================= C. STRATIFICATION (par blocs/tuiles) ===============
def stratification(masks, lon2, lat2, ds_t=None, ds_s=None):
    """Boucle temps (blocs) x tuiles. Retourne un dict de résultats (séries par région,
    cartes (période, saison), coupe à SECTION_LAT, histogrammes T-S par région/période/saison)."""
    if ds_t is None:
        ds_t, ds_s = open3d()
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
    names = diag_names()
    periods = {"P1": PERIOD1, "P2": PERIOD2}
    keys = [(p, s) for p in periods for s in MET_SEASONS]
    jrow = int(np.argmin(np.abs(lat - SECTION_LAT)))
    dz = layer_thickness(depth)
    nTS = (len(ts_axes()[0]) - 1) * (len(ts_axes()[1]) - 1)
    print(f"[strat] {nt} dates, {nz} niveaux (<= {MAX_DEPTH:.0f} m), {ny}x{nx} mailles, "
          f"tuiles {TILE}x{TILE}, blocs de {TIME_CHUNK} dates ; coupe à {lat[jrow]:.3f}N")

    reg_sum = {r: {n: np.zeros(nt) for n in names} for r in masks}
    reg_cnt = {r: {n: np.zeros(nt) for n in names} for r in masks}
    map_sum = {k: {n: np.zeros((ny, nx)) for n in names} for k in keys}
    map_cnt = {k: {n: np.zeros((ny, nx)) for n in names} for k in keys}
    sec = {k: {v: np.zeros((nz, nx)) for v in ("T", "S", "sig", "n")} for k in keys}
    ts_hist = {(r,) + k: np.zeros(nTS) for r in masks for k in keys}
    mfl = {r: m.astype(float) for r, m in masks.items()}
    wgt_all = np.cos(np.deg2rad(lat))[:, None]

    for a in range(0, nt, TIME_CHUNK):
        b = min(a + TIME_CHUNK, nt)
        sl = slice(int(idx[a]), int(idx[b - 1]) + 1, TIME_STRIDE)
        tc = time[idx[a:b]]
        th = ds_t["thetao"].isel(time=sl, depth=slice(0, nz)).values      # (t, z, y, x)
        sa = ds_s["so"].isel(time=sl, depth=slice(0, nz)).values
        tsel = {k: season_mask(tc, SEASONS[k[1]], periods[k[0]]) for k in keys}

        # coupe verticale (une latitude)
        th_r = th[:, :, jrow, :].astype("float64")
        sa_r = sa[:, :, jrow, :].astype("float64")
        sg_r = compute_sigma0(sa_r[:, :, None, :], th_r[:, :, None, :], depth,
                              lat[jrow:jrow + 1], lon)[:, :, 0, :]
        for k, m in tsel.items():
            if m.any():
                sec[k]["T"] += np.nansum(th_r[m], 0)
                sec[k]["S"] += np.nansum(sa_r[m], 0)
                sec[k]["sig"] += np.nansum(sg_r[m], 0)
                sec[k]["n"] += np.isfinite(th_r[m]).sum(0)

        for ys in range(0, ny, TILE):
            for xs in range(0, nx, TILE):
                ye, xe = min(ys + TILE, ny), min(xs + TILE, nx)
                th_t = th[:, :, ys:ye, xs:xe].astype("float64")
                if not np.isfinite(th_t[:, 0]).any():
                    continue                                              # tuile 100 % terre
                sa_t = sa[:, :, ys:ye, xs:xe].astype("float64")
                diags = strat_diagnostics(th_t, sa_t, depth, lat[ys:ye], lon[xs:xe])
                wgt = wgt_all[ys:ye]
                for r, m in mfl.items():
                    wm = m[ys:ye, xs:xe] * wgt
                    if not wm.any():
                        continue
                    for n, F in diags.items():
                        fin = np.isfinite(F)
                        reg_sum[r][n][a:b] += (np.where(fin, F, 0.0) * wm).sum(axis=(1, 2))
                        reg_cnt[r][n][a:b] += (fin * wm).sum(axis=(1, 2))
                    mloc = masks[r][ys:ye, xs:xe]                          # diagramme T-S
                    Tm, Sm = th_t[:, :, mloc], sa_t[:, :, mloc]            # (t, z, ncell)
                    for k, mt in tsel.items():
                        if mt.any():
                            add_ts(ts_hist[(r,) + k], Tm[mt], Sm[mt], dz)
                for k, mt in tsel.items():
                    if mt.any():
                        for n, F in diags.items():
                            Fm = F[mt]
                            map_sum[k][n][ys:ye, xs:xe] += np.nansum(Fm, axis=0)
                            map_cnt[k][n][ys:ye, xs:xe] += np.isfinite(Fm).sum(axis=0)
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
        section[k] = {v: np.where(c > 0, sec[k][v] / np.maximum(c, 1), np.nan)
                      for v in ("T", "S", "sig")}
    return dict(series=series, maps=maps, section=section, ts=ts_hist, lon=lon, lat=lat,
                depth=depth, jrow=jrow)


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


# ================================ LABELS (English) ==========================
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def diag_label(n):
    """(title, axis label with unit) for a variable."""
    if n.startswith("N2_"):
        return f"N² {n[3:-1]} m", "N² (s$^{-2}$)"
    if n.startswith("T_"):
        return f"Potential temperature {n[2:-1]} m", "θ (°C)"
    if n.startswith("S_"):
        return f"Salinity {n[2:-1]} m", "Salinity (PSS-78)"
    if n.startswith("z_sig"):
        return f"Depth of isopycnal σ₀ = {n[5:]}", "Depth (m)"
    return {"z_Smax": ("Depth of salinity maximum (LIW)", "Depth (m)"),
            "Smax": ("Maximum salinity (LIW)", "Salinity (PSS-78)"),
            "MLD": ("Mixed layer depth", "MLD (m)"),
            "MLD_max": ("Annual maximum MLD", "MLD (m)"),
            "w_mean": ("Mean Ekman vertical velocity w_E (signed)", "m/day"),
            "w_down": ("Downward Ekman pumping w_down (magnitude)", "m/day"),
            "tau_mag": ("Wind stress magnitude", "N m$^{-2}$"),
            "tau_y": ("Meridional wind stress", "N m$^{-2}$"),
            "frac_jours_forts": ("Fraction of strong-wind days", "fraction")}.get(n, (n, n))


def season_xlabel(season):
    return "Year (winter: year of January)" if season == "DJF" else "Year"


def season_title(season):
    return f"{season} ({'-'.join(MONTH_ABBR[m - 1] for m in SEASONS[season])})"


def key_diags():
    z, c = LAYERS[1], LAYERS[2]
    return [f"z_sig{ISOPYCNALS[1]}", f"N2_{z[0]}-{z[1]}m", f"T_{z[0]}-{z[1]}m",
            "Smax", "z_Smax", f"N2_{c[0]}-{c[1]}m"]


def _eddy_label():
    return f"Eddy region (EKE ≥ {EKE_ISOLINE:.0f} cm² s⁻²)"


def _control_label():
    return f"Control region (EKE < {CONTROL_EKE:.0f} cm² s⁻²)"


# ================================ FIGURES ===================================
def _outline(ax, lon, lat, mask, color, ls="-"):
    ax.contour(lon, lat, mask.astype(float), levels=[0.5], colors=color,
               linestyles=ls, linewidths=1.5)


def _mld_line(ax, lon, mld, lw=2.4):
    """MLD : white dashed line with a black outline (stays visible on any background)."""
    ax.plot(lon, mld, color="w", lw=lw, ls=(0, (4, 3)),
            path_effects=[pe.withStroke(linewidth=lw + 2.4, foreground="k")])


def plot_regions(eke_map, w_clim, masks, lon, lat, path):
    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    axs = axs.ravel()
    im = axs[0].pcolormesh(lon, lat, eke_map, cmap="viridis", shading="auto")
    plt.colorbar(im, ax=axs[0], label="Mean EKE (cm$^2$ s$^{-2}$)")
    axs[0].contour(lon, lat, eke_map, levels=[EKE_ISOLINE], colors="w", linewidths=1)
    _outline(axs[0], lon, lat, masks["eddy"], "r")
    _outline(axs[0], lon, lat, masks["control"], "k", "--")
    axs[0].legend(handles=[Line2D([], [], color="r", label=_eddy_label()),
                           Line2D([], [], color="k", ls="--", label=_control_label())],
                  loc="lower left", fontsize=8)
    axs[0].set_title("Mean EKE and analysis regions")
    v = np.nanpercentile(np.abs(np.concatenate([c.ravel() for c in w_clim.values()])), 98)
    for ax, s in zip(axs[1:], FORCING_SEASONS):
        im = ax.pcolormesh(lon, lat, w_clim[s], cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
        plt.colorbar(im, ax=ax, label="Mean w_E (m/day; >0 suction, <0 pumping)")
        _outline(ax, lon, lat, masks["eddy"], "k")
        ax.set_title(f"Ekman vertical velocity w_E, {season_title(s)}")
    for ax in axs:
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_forcing(annual_eddy, eke_ann, path):
    fig, axs = plt.subplots(2, 3, figsize=(18, 8.5))
    axs = axs.ravel()
    handles = None
    for ax, s in zip(axs, FORCING_SEASONS):
        df = annual_eddy[s]
        h1, = ax.plot(df.index, df["w_mean"], "o-", ms=3, color="tab:blue",
                      label="Mean w_E (signed; <0 = downward)")
        h2, = ax.plot(df.index, df["w_down"], "o-", ms=3, color="tab:cyan",
                      label="w_down: downward pumping magnitude (>0 = downward)")
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(2012.5, color="grey", ls=":")
        ax.set_xlabel(season_xlabel(s))
        ax.set_ylabel("w (m/day)")
        ax.set_title(f"Forcing {season_title(s)} vs EKE {EKE_SEASON}")
        ax2 = ax.twinx()
        h3, = ax2.plot(eke_ann.index, eke_ann.values, "s-", ms=3, color="tab:red", alpha=0.7,
                       label=f"EKE {EKE_SEASON} (eddy region)")
        ax2.set_ylabel("EKE (cm$^2$ s$^{-2}$)", color="tab:red")
        handles = [h1, h2, h3]
    axs[5].axis("off")
    axs[5].legend(handles=handles + [Line2D([], [], color="grey", ls=":", label="2012/2013 (P1/P2 split)")],
                  loc="center", fontsize=11, title="Legend")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_strat(annual, eke_ann, season, path):
    names = key_diags()
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
    fig.suptitle(f"Stratification, {season_title(season)}: eddy vs control region "
                 "(grey dotted line: 2012/2013 split)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_strat_maps(maps, masks, lon, lat, season, path):
    names = key_diags()[:4]
    fig, axs = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4.8), sharey=True)
    for ax, n in zip(np.atleast_1d(axs), names):
        d = maps[("P2", season)][n] - maps[("P1", season)][n]
        v = np.nanpercentile(np.abs(d), 98) if np.isfinite(d).any() else 1.0
        im = ax.pcolormesh(lon, lat, d, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
        _outline(ax, lon, lat, masks["eddy"], "k")
        title, ylab = diag_label(n)
        plt.colorbar(im, ax=ax, label=f"Δ {ylab}  (P2 − P1)")
        ax.set_title(f"{title}\n{season_title(season)}: {PERIOD2[0]}-{PERIOD2[1]} minus "
                     f"{PERIOD1[0]}-{PERIOD1[1]}", fontsize=8)
        ax.set_xlabel("Longitude (°E)")
    np.atleast_1d(axs)[0].set_ylabel("Latitude (°N)")
    np.atleast_1d(axs)[0].legend(handles=[Line2D([], [], color="k", label=_eddy_label())],
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
               Line2D([], [], color="w", lw=2.4, ls=(0, (4, 3)),
                      path_effects=[pe.withStroke(linewidth=4.8, foreground="k")],
                      label="Mean MLD (white dashed)"),
               Line2D([], [], color="k", ls="--", lw=1.2, label="Right column: P1 dashed / P2 solid"),
               Patch(color="tab:red", alpha=0.25, label=_eddy_label())]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10, frameon=True)
    fig.suptitle(f"Zonal section at {res['lat'][j]:.2f}°N: temperature and salinity (shading), "
                 f"isopycnals and MLD, {season_title(season)}")
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
    ax = axs[0]                                                   # (a) eddy vs control
    with np.errstate(divide="ignore"):
        dens = np.where(he > 0, np.log10(he), np.nan)
    im = ax.pcolormesh(Sc, Tc, dens, cmap="viridis", vmin=-5,
                       vmax=np.nanmax(dens) if np.isfinite(dens).any() else 0, shading="auto")
    plt.colorbar(im, ax=ax, label="log$_{10}$ (volume fraction), eddy region")
    contours(ax, hc, "tab:red")
    ax.legend(handles=[Line2D([], [], color="tab:red", label="Control region: contours enclosing 50% and 90% of its volume")],
              loc="upper left", fontsize=8)
    ax.set_title("Eddy region (colour) vs control region (red contours)", fontsize=10)
    ax = axs[1]                                                   # (b) difference
    d = smooth(he) - smooth(hc)
    v = np.nanpercentile(np.abs(d[d != 0]), 98) if np.any(d != 0) else 1.0
    im = ax.pcolormesh(Sc, Tc, d, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
    plt.colorbar(im, ax=ax, label="Δ volume fraction (eddy − control)")
    ax.set_title("Eddy minus control (red: more frequent in the eddy region)", fontsize=10)
    ax = axs[2]                                                   # (c) P1 vs P2
    contours(ax, H("eddy", ("P1",)), "tab:blue")
    contours(ax, H("eddy", ("P2",)), "tab:orange")
    ax.legend(handles=[Line2D([], [], color="tab:blue", label=f"Eddy region, P1 ({PERIOD1[0]}-{PERIOD1[1]})"),
                       Line2D([], [], color="tab:orange", label=f"Eddy region, P2 ({PERIOD2[0]}-{PERIOD2[1]})")],
              loc="upper left", fontsize=9, title="Contours: 50% and 90% of the volume")
    ax.set_title("Eddy region: P1 vs P2", fontsize=10)
    for ax in axs:                                                # isopycnals in the background
        cs = ax.contour(Sc, Tc, sig, levels=np.arange(22.0, 30.0, 0.5), colors="grey",
                        linewidths=0.6, linestyles=":")
        ax.clabel(cs, fmt="%.1f", fontsize=8)
        ax.set_xlabel("Salinity (PSS-78)")
    axs[0].set_ylabel("Potential temperature (°C)")
    occ = (he + hc) > 0                                           # axes fitted to the data
    if occ.any():
        ti, si = np.where(occ)
        axs[0].set_xlim(Sc[si.min()] - 0.1, Sc[si.max()] + 0.1)
        axs[0].set_ylim(Tc[ti.min()] - 0.5, Tc[ti.max()] + 0.5)
    fig.suptitle(f"θ-S diagrams, {season_title(season)}, 0-{MAX_DEPTH:.0f} m. Each bin = share of the region's "
                 "water volume (weighted by layer thickness). Dotted grey: isopycnals σ₀ (kg m$^{-3}$)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ================================== MAIN ====================================
VERSION = "v4 : figures en anglais, MLD en pointille blanc, labels isopycnes agrandis"


def main():
    print(f"[version] {VERSION}  |  fichier : {os.path.abspath(__file__)}")
    os.makedirs(OUTDIR, exist_ok=True)
    # --- A. EKE et régions
    vel = load_vg()
    lon, lat = vel["lon"], vel["lat"]
    eke_t, eke_map = compute_eke(vel)
    masks = build_masks(eke_map)
    eke_daily = pd.Series(region_mean(eke_t, masks["eddy"], lat), index=vel["time"])
    eke_ann = seasonal_mean(eke_daily, SEASONS[EKE_SEASON])
    eke_ann.to_csv(os.path.join(OUTDIR, f"eke_{EKE_SEASON}_region_eddy.csv"))
    np.savez(os.path.join(OUTDIR, "masques.npz"), eke_map=eke_map, lon=lon, lat=lat, **masks)
    del eke_t, vel

    # --- B. Forçage
    mflux = load_mflux()
    check_grid(lon, lat, mflux["lon"], mflux["lat"], "mflux")
    forcing, w = forcing_series(mflux, masks)
    t = mflux["time"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        w_clim = {s: np.nanmean(w[np.asarray(t.month.isin(SEASONS[s]))], axis=0)
                  for s in FORCING_SEASONS}
    plot_regions(eke_map, w_clim, masks, lon, lat, os.path.join(OUTDIR, "regions_et_w_Ekman.png"))
    mld = load_mld()
    check_grid(lon, lat, mld["lon"], mld["lat"], "mld")
    mld_ts = mld_series(mld, masks)

    # --- C. Stratification
    pkl = os.path.join(OUTDIR, "strat_results.pkl")
    if RECOMPUTE_STRAT or not os.path.exists(pkl):
        res = stratification(masks, lon, lat)
        pd.to_pickle(res, pkl)
    else:
        res = pd.read_pickle(pkl)
    j = res["jrow"]
    mld_sec = {}
    for p, per in (("P1", PERIOD1), ("P2", PERIOD2)):
        for s in MET_SEASONS:
            m = season_mask(mld["time"], SEASONS[s], per)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mld_sec[(p, s)] = np.nanmean(mld["mld"][m][:, j, :], axis=0)

    # --- D. Statistiques
    eddy_df = pd.concat([forcing, mld_ts["eddy"], res["series"]["eddy"]], axis=1)
    ctrl_df = pd.concat([mld_ts["control"], res["series"]["control"]], axis=1)
    all_seasons = list(dict.fromkeys(list(MET_SEASONS) + list(FORCING_SEASONS)))
    annual = {"eddy": build_annual(eddy_df, all_seasons), "control": build_annual(ctrl_df, all_seasons)}
    pd.set_option("display.width", 220)
    show = set(key_diags()) | {"w_mean", "w_down", "tau_mag", "frac_jours_forts", "MLD", "MLD_max"}
    for r in annual:
        tab = stats_table(annual[r], eke_ann)
        tab.to_csv(os.path.join(OUTDIR, f"stats_{r}.csv"), index=False)
        print(f"\n===== Région {r} : P1 {PERIOD1} vs P2 {PERIOD2} (variables clés ; tout dans stats_{r}.csv) =====")
        print(tab[tab["variable"].isin(show)].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        if r == "eddy" and len(tab):
            top = tab.reindex(tab["rho_EKE_sans_tendance"].abs().sort_values(ascending=False).index).head(10)
            print(f"\n--- 10 plus fortes corrélations avec l'EKE {EKE_SEASON} (sans tendance) ---")
            print(f"(attention : ~{len(tab)} tests, donc des corrélations fortes apparaissent par hasard)")
            print(top[["saison", "variable", "rho_EKE", "rho_EKE_sans_tendance",
                       "p_sans_tendance"]].to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    # --- Figures
    plot_forcing(annual["eddy"], eke_ann, os.path.join(OUTDIR, "forcage_saisons_vs_EKE.png"))
    for s in MET_SEASONS:
        plot_strat(annual, eke_ann, s, os.path.join(OUTDIR, f"stratification_series_{s}.png"))
        plot_strat_maps(res["maps"], masks, res["lon"], res["lat"], s,
                        os.path.join(OUTDIR, f"stratification_cartes_{s}.png"))
        plot_section(res, mld_sec, masks, s, os.path.join(OUTDIR, f"coupe_{SECTION_LAT}N_{s}.png"))
        plot_ts(res, s, os.path.join(OUTDIR, f"diagramme_TS_{s}.png"))
    print(f"\nSorties dans {OUTDIR}/")


if __name__ == "__main__":
    main()