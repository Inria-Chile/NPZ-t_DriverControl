import os
import numpy as np
import pandas as pd
import xarray as xr
import intake
import swifter
from multiprocessing import cpu_count
from xmip.preprocessing import combined_preprocessing
from npz_t_engine import apply_g_critic, apply_hyperplane

# --- Data processing functions ---
def setup_environment():
    
    if "ESMFMKFILE" not in os.environ:
        try:
            esmf_path = os.path.join(os.environ["HOME"], "git/esmf/lib/libO/")
            gfortran_dir = next(item for item in os.listdir(esmf_path) if "gfortran" in item)
            os.environ["ESMFMKFILE"] = os.path.join(esmf_path, gfortran_dir, "esmf.mk")
            print(f"✅ ESMFMKFILE configured in: {os.environ['ESMFMKFILE']}")
        except (StopIteration, KeyError):
            print("⚠️ Check ~/git/esmf/")

# --- LÓGICA DE PROCESAMIENTO ---

def get_cmip6_datatree(
    source_ids=["GFDL-ESM4"], 
    variable_ids=["tos", "mlotst"], 
    experiment_ids=["ssp585","historical"],
    table_id="Omon",
    grid_label="gn"
):

    print(f"🔍 Searching in Pangeo: {source_ids} | {experiment_ids}...")
    
    col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    
    cat = col.search(
        source_id=source_ids,
        variable_id=variable_ids,
        table_id=table_id,
        grid_label=grid_label,
        experiment_id=experiment_ids,
        require_all_on=["source_id"],
    )

    if len(cat) == 0:
        raise ValueError("❌ There are no values")

    kwargs = dict(
        preprocess=combined_preprocessing,
        xarray_open_kwargs=dict(use_cftime=True),
        storage_options={"token": "anon"},
        join="outer",
        compat="no_conflicts",
    )
    
    cat.esmcat.aggregation_control.groupby_attrs = ["source_id", "experiment_id"]
    
    return cat.to_datatree(**kwargs)

def load_dataframe(model_name, scenario_name, start_slice, end_slice):

    ssp585 = dt[model_name][scenario_name].ds

    da_temp = ssp585["tos"].sel(time=slice(start_slice, end_slice))
    da_mld = ssp585["mlotst"].sel(time=slice(start_slice, end_slice))

    df_temp = da_temp.to_dataframe(name="tos").reset_index()
    df_mld = da_mld.to_dataframe(name="mlotst").reset_index()

    df_temp["month"] = [t.month for t in df_temp["time"]]
    df_mld["month"] = [t.month for t in df_mld["time"]]

    df_temp_monthly = (
        df_temp.groupby(["lat", "lon", "month"])["tos"]
        .mean()
        .reset_index()
        .sort_values(by=["lat", "lon", "month"]) 
    )

    df_mld_monthly = (
        df_mld.groupby(["lat", "lon", "month"])["mlotst"]
        .mean()
        .reset_index()
        .sort_values(by=["lat", "lon", "month"]) 
    )

    df_temp_vec = df_temp_monthly.groupby(["lat", "lon"])["tos"].apply(lambda v: v.tolist()).reset_index()

    df_mld_vec = df_mld_monthly.groupby(["lat", "lon"])["mlotst"].apply(lambda v: v.tolist()).reset_index()

    df_temp_vec = df_temp_vec.rename(columns={"tos": "temperature_vector"})
    df_mld_vec = df_mld_vec.rename(columns={"mlotst": "mld_vector"})

    df_merged = df_temp_vec.merge(df_mld_vec, on=["lat", "lon"], how="inner")
    df_merged["model_name"] = model_name
    df_merged["scenario_name"] = scenario_name

    return df_merged
def load_hybrid(model_name, scenario_name, target_year, member_idx=0):
    """
    Carga datos híbridos para un año específico:
    1. HISTÓRICO (Fijo 1960-2000): Promedios climáticos base.
    2. TARGET (target_year): Vectores mensuales para ese año específico.
    """
    
    ds_full = dt[model_name][scenario_name].ds
    ds_hist = dt[model_name]["historical"].ds

    ds_hist=ds_hist.isel(member_id=member_idx)
    
    ds_hist = ds_hist.sel(time=slice("1960", "2000"))
    
    da_temp_base = ds_hist["tos"].mean(dim="time", keep_attrs=True)
    da_mld_base = ds_hist["mlotst"].groupby("time.year").max().mean(dim="year", keep_attrs=True)

    ds_target = ds_full.sel(time=str(target_year))
    
    if ds_target.sizes['time'] != 12:
        print(f"⚠️ Advertencia: El año {target_year} tiene {ds_target.sizes['time']} pasos de tiempo (se esperaban 12).")

    vals_temp_target = ds_target["tos"].values
    vals_mld_target = ds_target["mlotst"].values

    lat_2d, lon_2d, temp_base_2d, mld_base_2d = xr.broadcast(
        ds_full.lat, ds_full.lon, da_temp_base, da_mld_base
    )
    
    lat_flat = lat_2d.values.flatten()
    lon_flat = lon_2d.values.flatten()
    temp_base_flat = temp_base_2d.values.flatten()
    mld_base_flat = mld_base_2d.values.flatten()
    
    temp_vec_flat = vals_temp_target.reshape(12, -1).T
    mld_vec_flat = vals_mld_target.reshape(12, -1).T

    if len(lat_flat) != len(temp_vec_flat):
        raise ValueError(f"Desajuste de grilla: Coordenadas {len(lat_flat)} vs Datos {len(temp_vec_flat)}")

    df = pd.DataFrame({
        "lat": lat_flat,
        "lon": lon_flat,
        
        "temp_mean_hist": temp_base_flat,
        "mld_max_hist": mld_base_flat,
        
        "temp_vec_year": list(temp_vec_flat),
        "mld_vec_year": list(mld_vec_flat)
    })
    
    df["model"] = model_name
    df["scenario"] = scenario_name
    df["year_data"] = target_year

    return df

def mask_ice_with_nan_list(cell, threshold=-1.8):
    nan_list = [np.nan] * 12
    try:
        if isinstance(cell, list) and len(cell) > 0:
            arr = cell
        else:
            return nan_list

        if np.all(np.isnan(arr)) or np.nanmin(arr) <= threshold:
            return nan_list
        
        return arr 
    
    except:
        return nan_list
def clean_data(df):
    df['temperature_vector'] = df['temperature_vector'].apply(mask_ice_with_nan_list)
    df['mld_vector'] = df['mld_vector'].apply(mask_ice_with_nan_list)
    return df

def compute_g_critic(df,year="2024", model="GFDL-ESM4", scenario="ssp585"):
    df = clean_data(df)
    df = apply_g_critic(df)
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df.to_parquet(os.path.join(output_dir, f"g_critic_{year}_{scenario}_{model}.parquet"))

def compute_hyperplane(df):
    df = clean_data(df)
    df = apply_hyperplane(df)
    return df

def calculate_mean_temp(row):
    temp_vec = row["temp_vec_year"]
    if isinstance(temp_vec, (list, np.ndarray)):
        return np.mean(temp_vec)
    return np.nan

def calculate_max_mld(row):
    mld_vec = row["mld_vec_year"]
    if isinstance(mld_vec, (list, np.ndarray)):
        return np.max(mld_vec)
    return np.nan

def data_merged(df_hyper,df_hybrid,year="2024",scenario="ssp585",model="GFDL-ESM4"):
    df_hybrid['lat'] = df_hybrid['lat'].astype('float64')
    df_hybrid['lon'] = df_hybrid['lon'].astype('float64')
    df_merged = pd.merge(df_hyper, df_hybrid, on=['lat', 'lon'], how='inner')
    df_merged["temp_mean_year"] = df_merged.apply(calculate_mean_temp, axis=1)
    df_merged["mld_max_year"] = df_merged.apply(calculate_max_mld, axis=1)
    df_merged["sens_a"] = df_merged["S_temp"] * (df_merged["temp_mean_year"]- df_merged["temp_mean_hist"])
    df_merged["sens_b"] = df_merged["S_mld"] * (df_merged["mld_max_year"] - df_merged["mld_max_hist"])
    df_merged["sens"] = df_merged["sens_a"].abs()/(df_merged["sens_a"].abs() + df_merged["sens_b"].abs())
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df_merged.to_parquet(os.path.join(output_dir, f"hyperplane_{year}_{scenario}_{model}.parquet"))


# --- Principal execution block---
if __name__ == "__main__":

    setup_environment()

    dt = get_cmip6_datatree()

    df_2024_g = load_dataframe("GFDL-ESM4", "ssp585", "2024-01", "2024-12")

    df_2100_g = load_dataframe("GFDL-ESM4", "ssp585", "2100-01", "2100-12")

    df_2024_h = load_hybrid("GFDL-ESM4", "ssp585", "2024")

    df_2100_h = load_hybrid("GFDL-ESM4", "ssp585", "2100")

    compute_g_critic(df_2024_g)

    compute_g_critic(df_2100_g, year="2100")

    df_2024_g = compute_hyperplane(df_2024_g)

    df_2100_g = compute_hyperplane(df_2100_g)

    df_2024_g = pd.read_parquet("/home/matiasneto/git/NPZ-T_Persistence/data/hyperplane_2024_ssp585_GFDL-ESM4.parquet")

    df_2100_g = pd.read_parquet("/home/matiasneto/git/NPZ-T_Persistence/data/hyperplane_2100_ssp585_GFDL-ESM4.parquet")

    data_merged(df_2024_g, df_2024_h)

    data_merged(df_2100_g, df_2100_h, year="2100")





