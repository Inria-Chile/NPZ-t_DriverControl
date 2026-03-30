import os
import numpy as np
import pandas as pd
import xarray as xr
import intake
import swifter
from multiprocessing import cpu_count
from xmip.preprocessing import combined_preprocessing
from npz_t_engine import apply_g_critic

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
    experiment_ids=["ssp585"],
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

    # Verificación de seguridad: ¿Encontró algo?
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

def mask_ice_with_nan_list(cell, threshold=-1.8):
    # 1. Creamos la lista de "emergencia" con 12 NaNs
    nan_list = [np.nan] * 12
    
    try:
        # Extraer el contenido (manejando tu estructura de lista de arrays)
        if isinstance(cell, list) and len(cell) > 0:
            arr = cell
        else:
            return nan_list

        # Si es Tierra (todo NaN) o Hielo (min <= threshold)
        if np.all(np.isnan(arr)) or np.nanmin(arr) <= threshold:
            return nan_list
        
        # Si es océano abierto, devolvemos el array original (asegurando que sea lista/array de 12)
        return arr 
    
    except:
        # Ante cualquier duda o error (como listas vacías), devolvemos los 12 NaNs
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
    return df

# --- Principal execution block---
if __name__ == "__main__":

    setup_environment()

    dt = get_cmip6_datatree()

    df_2024_g = load_dataframe("GFDL-ESM4", "ssp585", "2024-01", "2024-12")
    df_2100_g = load_dataframe("GFDL-ESM4", "ssp585", "2100-01", "2100-12")

    df_2024_g = compute_g_critic(df_2024_g)

    df_2100_g = compute_g_critic(df_2100_g, year="2100")