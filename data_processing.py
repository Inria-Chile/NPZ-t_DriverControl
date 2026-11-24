import os
import subprocess
import platform
import numpy as np
import pandas as pd
import xarray as xr
import intake
import swifter
from multiprocessing import cpu_count
from xmip.preprocessing import combined_preprocessing
from npz_t_engine import apply_g_critic, apply_hyperplane_3d, fit_light_vector, data_merged

def setup_environment():
    if "ESMFMKFILE" in os.environ:
        print(f"✅ ESMFMKFILE already set: {os.environ['ESMFMKFILE']}")
        return

    try:
        path_check = subprocess.check_output(["which", "esmf_regrid"], text=True).strip()
        if path_check:
            base_path = os.path.dirname(os.path.dirname(path_check))
            potential_mk = os.path.join(base_path, "lib", "esmf.mk")
            if os.path.exists(potential_mk):
                os.environ["ESMFMKFILE"] = potential_mk
                print(f"✅ ESMFMKFILE found via system: {os.environ['ESMFMKFILE']}")
                return
    except Exception:
        pass

    try:
        home = os.environ["HOME"]
        search_path = os.path.join(home, "git/esmf/lib/libO/")
        
        if os.path.exists(search_path):
            subdirs = os.listdir(search_path)
            if subdirs:
                selected_dir = subdirs[0]
                os.environ["ESMFMKFILE"] = os.path.join(search_path, selected_dir, "esmf.mk")
                print(f"✅ ESMFMKFILE found in git folder: {os.environ['ESMFMKFILE']}")
                return
    except Exception:
        pass


# --- Process Logic ---

def get_cmip6_datatree(
    source_ids=["GFDL-ESM4"], 
    variable_ids=["tos", "mlotst", "rsds"], 
    experiment_ids=["ssp585", "historical"],
    table_ids=["Omon", "Amon"],       # Limitado a las tablas mensuales verificadas
    grid_labels=["gn", "gr1"]         # 'gn' para el océano limpio, 'gr1' para la luz
):

    print(f"🔍 Searching verified CMIP6 paths in Pangeo...")
    
    col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    
    # Buscamos las variables cruzando los filtros identificados
    cat = col.search(
        source_id=source_ids,
        variable_id=variable_ids,
        table_id=table_ids,
        grid_label=grid_labels,
        experiment_id=experiment_ids
        # Eliminamos require_all_on para evitar el bloqueo por desajuste de grilla
    )

    if len(cat) == 0:
        raise ValueError("❌ Error inesperado: Catálogo vacío.")

    kwargs = dict(
        preprocess=combined_preprocessing,
        xarray_open_kwargs=dict(use_cftime=True),
        storage_options={"token": "anon"},
        join="outer",
        compat="no_conflicts",
    )
    
    # CLAVE: Agrupamos incluyendo tabla y grilla para mantener el océano (gn) 
    # y la atmósfera (gr1) en ramas independientes y limpias del DataTree
    cat.esmcat.aggregation_control.groupby_attrs = ["source_id", "experiment_id", "table_id", "grid_label"]
    
    print("⏳ Loading data into DataTree... (Esto puede tomar un par de minutos)")
    return cat.to_datatree(**kwargs)

def load_dataframe(model_name, scenario_name, start_slice, end_slice):
    print(f"📦 Extracting and aligning vectors for {model_name} ({scenario_name})...")
    
    # 1. Acceder a los nodos correspondientes del DataTree
    ds_ocean = dt[model_name][scenario_name]['Omon']['gn'].ds
    ds_atmos = dt[model_name][scenario_name]['Amon']['gr1'].ds

    # 2. Cortar la ventana temporal de interés
    ds_ocean_sliced = ds_ocean[["tos", "mlotst"]].sel(time=slice(start_slice, end_slice))
    da_light_sliced = ds_atmos["rsds"].sel(time=slice(start_slice, end_slice))

    # 3. Calcular la climatología mensual (12 meses) en Xarray (¡Ultra rápido!)
    print("  ↳ Computing monthly seasonal cycles in Xarray...")
    clim_ocean = ds_ocean_sliced.groupby("time.month").mean("time")
    clim_light = da_light_sliced.groupby("time.month").mean("time")

    # 4. REMALLADO: Limpiamos los nombres conflictivos y renombramos directo
    print("  ↳ Regridding light data to match ocean grid geometry...")
    
    if "y" in clim_light.dims and "x" in clim_light.dims:
        # Borramos las variables 'lat' y 'lon' que causan el conflicto de nombres
        clim_light = clim_light.drop_vars(["lat", "lon"], errors="ignore")
        # Ahora sí podemos renombrar los ejes y y x sin que nadie proteste
        clim_light = clim_light.rename({"y": "lat", "x": "lon"})

    clim_light_regridded = clim_light.interp(
        lat=clim_ocean["lat"], 
        lon=clim_ocean["lon"], 
        method="linear"
    )

    # 5. Combinar todo en un único Dataset de Xarray
    clim_combined = clim_ocean.assign(rsds=clim_light_regridded)

    # 6. Convertir a DataFrame (ahora es una matriz pequeña de solo 12 meses por pixel)
    print("  ↳ Converting filtered fields to Pandas...")
    df_monthly = clim_combined.to_dataframe().reset_index()

    # Asegurar el orden cronológico de los meses (1 al 12) para cada coordenada
    df_monthly = df_monthly.sort_values(by=["lat", "lon", "month"])

    # 7. Vectorización: Construir los vectores de 12 elementos por celda
    print("  ↳ Packing time series into 12-month analytical vectors...")
    df_vectors = (
        df_monthly.groupby(["lat", "lon"])
        .agg({
            "tos": lambda v: v.tolist(),
            "mlotst": lambda v: v.tolist(),
            "rsds": lambda v: v.tolist()  # <-- Tu nuevo vector de luz estacional
        })
        .reset_index()
    )

    # Renombrar columnas para mantener consistencia con tu solucionador de Floquet
    df_vectors = df_vectors.rename(
        columns={
            "tos": "temperature_vector",
            "mlotst": "mld_vector",
            "rsds": "light_vector"
        }
    )

    # Inyectar Metadatos descriptivos
    df_vectors["model_name"] = model_name
    df_vectors["scenario_name"] = scenario_name

    print("✅ Extraction and alignment complete!")
    return df_vectors
def load_hybrid(model_name, scenario_name, target_year, member_idx=0):
    print(f"📦 Extracting hybrid baseline and target vectors for {model_name}...")
    
    # 1. Acceder a los conjuntos de datos de océano y atmósfera
    ds_ocean_full = dt[model_name][scenario_name]['Omon']['gn'].ds
    ds_atmos_full = dt[model_name][scenario_name]['Amon']['gr1'].ds
    
    ds_ocean_hist = dt[model_name]['historical']['Omon']['gn'].ds
    ds_atmos_hist = dt[model_name]['historical']['Amon']['gr1'].ds

    # Control de miembros del ensamble
    if "member_id" in ds_ocean_hist.dims: ds_ocean_hist = ds_ocean_hist.isel(member_id=member_idx)
    if "member_id" in ds_ocean_full.dims: ds_ocean_full = ds_ocean_full.isel(member_id=member_idx)
    if "member_id" in ds_atmos_full.dims: ds_atmos_full = ds_atmos_full.isel(member_id=member_idx)
    if "member_id" in ds_atmos_hist.dims: ds_atmos_hist = ds_atmos_hist.isel(member_id=member_idx)

    # 2. Cortar y calcular las líneas de base históricas (1960-2000)
    print("  ↳ Computing historical baselines (1960-2000)...")
    ds_ocean_hist_sliced = ds_ocean_hist.sel(time=slice("1960", "2000"))
    ds_atmos_hist_sliced = ds_atmos_hist.sel(time=slice("1960", "2000"))
    
    da_temp_base = ds_ocean_hist_sliced["tos"].mean(dim="time", keep_attrs=True)
    da_mld_base = ds_ocean_hist_sliced["mlotst"].groupby("time.year").max().mean(dim="year", keep_attrs=True)
    
    # CLIMATOLOGÍA HISTÓRICA DE LA LUZ (12 meses reales del pasado)
    clim_light_hist = ds_atmos_hist_sliced["rsds"].groupby("time.month").mean("time")

    # 3. Cortar el año objetivo futuro
    print(f"  ↳ Slicing target year: {target_year}...")
    ds_ocean_target = ds_ocean_full.sel(time=str(target_year))
    ds_atmos_target = ds_atmos_full.sel(time=str(target_year))

    clim_ocean_target = ds_ocean_target[["tos", "mlotst"]].groupby("time.month").mean("time")
    clim_light_target = ds_atmos_target["rsds"].groupby("time.month").mean("time")

   # 4. REMALLADO: Alinear grillas de atmósfera (futura e histórica) al océano
    print("  ↳ Regridding historical and target light data...")
    
    # Modificación explícita para la luz del Futuro
    if "y" in clim_light_target.dims and "x" in clim_light_target.dims:
        clim_light_target = clim_light_target.drop_vars(["lat", "lon"], errors="ignore").rename({"y": "lat", "x": "lon"})
        
    # Modificación explícita para la luz del Pasado (Climatología Histórica)
    if "y" in clim_light_hist.dims and "x" in clim_light_hist.dims:
        clim_light_hist = clim_light_hist.drop_vars(["lat", "lon"], errors="ignore").rename({"y": "lat", "x": "lon"})

    # Ahora sí interpolará sin chistar porque las dimensiones existen de verdad
    clim_light_target_regridded = clim_light_target.interp(
        lat=clim_ocean_target["lat"], 
        lon=clim_ocean_target["lon"], 
        method="linear"
    )
    clim_light_hist_regridded = clim_light_hist.interp(
        lat=clim_ocean_target["lat"], 
        lon=clim_ocean_target["lon"], 
        method="linear"
    )

    # 5. Combinar todo en un único Dataset de Xarray
    clim_combined = clim_ocean_target.assign(
        rsds=clim_light_target_regridded,
        rsds_hist=clim_light_hist_regridded,  
        temp_mean_hist=da_temp_base,
        mld_max_hist=da_mld_base
    )

    # 6. Conversión estructurada a Pandas
    df_monthly = clim_combined.to_dataframe().reset_index()
    df_monthly = df_monthly.sort_values(by=["lat", "lon", "month"])

    # 7. Vectorización por píxel
    print("  ↳ Packing seasonal profiles into analytical arrays...")
    df_vectors = (
        df_monthly.groupby(["lat", "lon"])
        .agg({
            "temp_mean_hist": "first",
            "mld_max_hist": "first",
            "tos": lambda v: v.tolist(),
            "mlotst": lambda v: v.tolist(),
            "rsds": lambda v: v.tolist(),
            "rsds_hist": lambda v: v.tolist()  # <-- Vector histórico a lista
        })
        .reset_index()
    )

    # Renombrar columnas para el resolvedor
    df_vectors = df_vectors.rename(
        columns={
            "tos": "temperature_vector",
            "mlotst": "mld_vector",
            "rsds": "light_vector",
            "rsds_hist": "light_vector_hist"  # <-- Nombre definitivo de la luz pasada
        }
    )
    
    df_vectors["model"] = model_name
    df_vectors["scenario"] = scenario_name
    df_vectors["year_data"] = target_year

    print(f"✅ Hybrid dataset complete for the year {target_year}!")
    return df_vectors

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
    df['light_vector'] = df['light_vector'].apply(mask_ice_with_nan_list)
    return df

def compute_g_critic(df,year="2024", model="GFDL-ESM4", scenario="ssp585"):
    df = clean_data(df)
    df = apply_g_critic(df)
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df.to_parquet(os.path.join(output_dir, f"g_critic_{year}_{scenario}_{model}.parquet"))

def compute_hyperplane(df,year="2024", model="GFDL-ESM4", scenario="ssp585"):
    df = clean_data(df)
    df = apply_hyperplane_3d(df)
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df.to_parquet(os.path.join(output_dir, f"hyperplane_{year}_{scenario}_{model}.parquet"))
    return df

def compute_realized_impacts(df, year="2024",scenario="ssp585",model="GFDL-ESM4"):
    print("🧠 Deconstructing 3D climate anomalies and ecosystem impacts from real data...")
    
    t_mean_target = []
    mld_max_target = []
    light_amp_target = []
    light_amp_historical = []
    
    # Extraer los parámetros geométricos de los arrays reales de cada píxel
    for idx, row in df.iterrows():
        if pd.isna(row["temp_mean_hist"]):  # Ignorar celdas de tierra
            t_mean_target.append(np.nan)
            mld_max_target.append(np.nan)
            light_amp_target.append(np.nan)
            light_amp_historical.append(np.nan)
            continue
            
        # Parámetros del año objetivo (Futuro)
        t_mean_target.append(np.mean(row["temperature_vector"]))
        mld_max_target.append(np.max(row["mld_vector"]))
        
        # Amplitud real del futuro vía regresión armónica
        _, i_amp_fut, _ , _ , _ = fit_light_vector(row["light_vector"])
        light_amp_target.append(i_amp_fut)
        
        # Amplitud real del pasado (1960-2000) vía regresión armónica (¡Sin inventar nada!)
        _, i_amp_hist, _ , _ , _ = fit_light_vector(row["light_vector_hist"])
        light_amp_historical.append(i_amp_hist)

    # Asignar los rasgos calculados de vuelta al DataFrame
    df["temp_mean_target"] = t_mean_target
    df["mld_max_target"] = mld_max_target
    df["light_amp_target"] = light_amp_target
    df["light_amp_hist"] = light_amp_historical

    # CÁLCULO DE ANOMALÍAS CLIMÁTICAS DETERMINISTAS
    df["delta_T"] = df["temp_mean_target"] - df["temp_mean_hist"]
    df["delta_H"] = df["mld_max_target"] - df["mld_max_hist"]
    df["delta_I_amp"] = df["light_amp_target"] - df["light_amp_hist"]  # Anomalía real de nubes/radiación

    # CÁLCULO DE IMPACTOS REALIZADOS (I = |S| * |Anomalía|)
    df["Impact_Temp"] = np.abs(df["S_temp"]) * np.abs(df["delta_T"])
    df["Impact_Mld"] = np.abs(df["S_mld"]) * np.abs(df["delta_H"])
    df["Impact_Light"] = np.abs(df["S_light"]) * np.abs(df["delta_I_amp"])

    # DETERMINACIÓN DEL ÍNDICE DE DOMINANCIA METABÓLICA 3D (M_T)
    total_impact = df["Impact_Temp"] + df["Impact_Mld"] + df["Impact_Light"]
    df["M_T"] = df["Impact_Temp"] / total_impact

    print("✅ Climate anomalies and metabolic dominance index (M_T) unraveled successfully!")
    output_dir = "data1"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df.to_parquet(os.path.join(output_dir, f"hyperplane_{year}_{scenario}_{model}.parquet"))


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

    #df_2024_g = pd.read_parquet("~/data/hyperplane_2024_ssp585_GFDL-ESM4.parquet")

    #df_2100_g = pd.read_parquet("~/data/hyperplane_2100_ssp585_GFDL-ESM4.parquet")

    compute_realized_impacts(df_2024_h)
    compute_realized_impacts(df_2100_h, year="2100")





