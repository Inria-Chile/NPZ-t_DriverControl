import os
import numpy as np
import pandas as pd
import xarray as xr
import intake
import intake_esm
from multiprocessing import cpu_count
from xmip.preprocessing import combined_preprocessing

# --- CONFIGURACIÓN DE ENTORNO (Lo que antes era manual) ---
def setup_environment():
    """Configura las variables de sistema para ESMF"""
    
    # Localizar ESMFMKFILE automáticamente
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
    experiment_ids=["ssp585", "historical"],
    table_id="Omon",
    grid_label="gn"
):
    """
    Conecta a Pangeo y devuelve el datatree filtrado por parámetros.
    """
    print(f"🔍 Buscando en Pangeo: {source_ids} | {experiment_ids}...")
    
    col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    
    # Buscamos usando las listas pasadas como argumentos
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
        raise ValueError("❌ No se encontraron datos para los criterios especificados.")

    kwargs = dict(
        preprocess=combined_preprocessing,
        xarray_open_kwargs=dict(use_cftime=True),
        storage_options={"token": "anon"},
        join="outer",
        compat="no_conflicts",
    )
    
    # Agrupamos por los atributos que definen la jerarquía del DataTree
    cat.esmcat.aggregation_control.groupby_attrs = ["source_id", "experiment_id"]
    
    print("🌳 Construyendo DataTree (esto puede tardar un poco)...")
    return cat.to_datatree(**kwargs)


# --- BLOQUE DE EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    # 1. Preparar sistema
    setup_environment()
    # 3. Descarga y Procesamiento
    print("📥 Conectando a Pangeo...")
    dt = get_cmip6_datatree()
