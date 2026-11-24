
import pandas as pd
import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import seaborn as sns
import cartopy.feature as cfeature
import numpy as np
from scipy.interpolate import griddata
from data_processing import clean_data

def fix_longitude(lon):
    lon_fixed = ((lon + 180) % 360) - 180
    return lon_fixed
def plot_g_critic(df, year="2024", model="GFDL-ESM4", zoom_spg=False):
# 1. Preparar datos y purgar la grieta artificial de Greenwich (entre -1° y 1.5°)
    # 1. Preparar datos y purgar la grieta artificial de Greenwich (entre -1° y 1.5°)
    df_ice_base = df.copy()
    lon_fixed = (df_ice_base["lon"].to_numpy() + 180) % 360 - 180
    
    # Filtrar NaNs artificiales de la grieta en el mapa base
    is_fake_nan = (lon_fixed >= -1.0) & (lon_fixed <= 1.5) & df_ice_base["g_critic"].isna()
    df_ice_base = df_ice_base[~is_fake_nan]
    
    # Crear el set de datos 100% limpio (sin ningún NaN) para interpolar el océano continuo
    df_clean = df_ice_base.dropna(subset=["g_critic"])
    
    # Coordenadas limpias del océano
    lon_cl = (df_clean["lon"].to_numpy() + 180) % 360 - 180
    lat_cl = df_clean["lat"].to_numpy()
    g_v = df_clean["g_critic"].to_numpy()
    
    # Crear indicador binario de datos válidos en el mapa base (1 = océano, 0 = tierra/hielo)
    lon_raw = (df_ice_base["lon"].to_numpy() + 180) % 360 - 180
    lat_raw = df_ice_base["lat"].to_numpy()
    base_var = df_ice_base["g_critic"].to_numpy()
    valid_indicator = (~np.isnan(base_var)).astype(float)
    
    # 2. Acolchado Cíclico en extremos del Pacífico (+/-150°) para evitar cortes laterales
    pad_l = lon_cl < -150
    pad_r = lon_cl > 150
    lon_cl_p = np.concatenate([lon_cl, lon_cl[pad_l] + 360, lon_cl[pad_r] - 360])
    lat_cl_p = np.concatenate([lat_cl, lat_cl[pad_l], lat_cl[pad_r]])
    val_cl_p = np.concatenate([g_v, g_v[pad_l], g_v[pad_r]])
    
    pad_raw_l = lon_raw < -150
    pad_raw_r = lon_raw > 150
    lon_raw_p = np.concatenate([lon_raw, lon_raw[pad_raw_l] + 360, lon_raw[pad_raw_r] - 360])
    lat_raw_p = np.concatenate([lat_raw, lat_raw[pad_raw_l], lat_raw[pad_raw_r]])
    valid_p = np.concatenate([valid_indicator, valid_indicator[pad_raw_l], valid_indicator[pad_raw_r]])
    
    # 3. Crear grilla y realizar la interpolación dual según el zoom
    if zoom_spg:
        # Grilla regional optimizada para el Giro Subpolar
        lon_grid = np.linspace(-90, 10, 1000)
        lat_grid = np.linspace(25, 75, 1000)
        lon_mesh_final, lat_mesh_final = np.meshgrid(lon_grid, lat_grid)
        
        # Interpolación limpia del océano
        g_interp_linear = griddata((lon_cl_p, lat_cl_p), val_cl_p, (lon_mesh_final, lat_mesh_final), method="linear")
        g_interp_nearest = griddata((lon_cl_p, lat_cl_p), val_cl_p, (lon_mesh_final, lat_mesh_final), method="nearest")
        g_interp_clean = np.where(np.isnan(g_interp_linear), g_interp_nearest, g_interp_linear)
        
        # Interpolación de la máscara binaria para suavizar bordes
        grid_mask = griddata((lon_raw_p, lat_raw_p), valid_p, (lon_mesh_final, lat_mesh_final), method="linear")
        
        # Aplicamos la máscara con umbral de tolerancia para permitir el recorte perfecto de Cartopy
        mask_condition = (grid_mask < 0.4) | np.isnan(grid_mask)
        g_interp_clean[mask_condition] = np.nan
        g_interp_final = g_interp_clean
    else:
        # Grilla global completa
        lon_grid = np.linspace(-180, 180, 1200)
        lat_grid = np.linspace(-90, 90, 1200)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        # Interpolación limpia global del océano
        g_interp_linear = griddata((lon_cl_p, lat_cl_p), val_cl_p, (lon_mesh, lat_mesh), method="linear")
        g_interp_nearest = griddata((lon_cl_p, lat_cl_p), val_cl_p, (lon_mesh, lat_mesh), method="nearest")
        g_interp_clean = np.where(np.isnan(g_interp_linear), g_interp_nearest, g_interp_linear)
        
        # Interpolación de la máscara binaria global
        grid_mask = griddata((lon_raw_p, lat_raw_p), valid_p, (lon_mesh, lat_mesh), method="linear")
        
        mask_condition = (grid_mask < 0.4) | np.isnan(grid_mask)
        g_interp_clean[mask_condition] = np.nan
        
        # Sello cíclico manual para cerrar herméticamente el mapa global
        lon_step = lon_grid[1] - lon_grid[0]
        lon_grid_cycl = np.append(lon_grid, lon_grid[-1] + lon_step)
        g_interp_final = np.hstack([g_interp_clean, g_interp_clean[:, :1]])
        lon_mesh_final, lat_mesh_final = np.meshgrid(lon_grid_cycl, lat_grid)
    
# 4. Configurar la escena con Cartopy
    fig = plt.figure(figsize=(12, 6), dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Teñimos el fondo del mapa (hielo y NaNs profundos) con un gris azulado frío pálido
    ax.set_facecolor('#cbd5e1') 
    
    # Tierra firme en gris medio para un contraste limpio
    ax.add_feature(cfeature.LAND, facecolor="#94a3b8", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    
    if zoom_spg:
        ax.set_extent([-80, 0, 35, 65], crs=ccrs.PlateCarree())
        
    vmin, vmax = 0.0, 2.0
    levels = np.linspace(vmin, vmax, 41) 
    
    # Graficar usando transform_first=True
    im = ax.contourf(
        lon_mesh_final, lat_mesh_final, g_interp_final,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r", 
        levels=levels,
        extend='both',
        zorder=1,
        transform_first=True
    )
    im.set_rasterized(True)
    
    # 5. Colorbar destacando el 1.0 en el centro
    cbar = plt.colorbar(im, orientation="vertical", shrink=0.7, pad=0.05)
    
    # AJUSTES DE FUENTE PARA CONSISTENCIA
    cbar.set_label(fr"$\gamma^{{{year}}}$", fontsize=11, labelpad=10) # Igual que en el mapa categórico
    cbar.ax.tick_params(labelsize=10) # Tamaño 10 para los números, igual que los labels categóricos
    
    cbar.solids.set_rasterized(True) 
    cbar.solids.set_edgecolor("face")
    cbar.set_ticks([0.0, 0.5, 1.0, 1.5, 2.0])


    if zoom_spg:
        plt.title(f"SPG Critical Nutrient Saturation Ratio ({year},{model})")
        output_dir = "figures"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(f"figures/g_critic_{year}_{model}_spg.pdf", bbox_inches='tight', dpi=300)
    else:
        plt.title(f"Global Critical Nutrient Saturation ({year},{model})")
        output_dir = "figures"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(f"figures/g_critic_{year}_{model}.pdf", bbox_inches='tight', dpi=300)
    #plt.show()

def plot_driver_dominance(df, year="2024", zoom_spg=False):
    """
    Genera el mapa discreto con contornos suaves (contourf).
    Mantiene el calado de hielo orgánico de g_critic pero purga la grieta
    de NaNs de Greenwich para eliminar la línea blanca de forma definitiva.
    """
    # 1. Calcular los drivers dominantes usando el dataset limpio
    df_clean = clean_data(df)
    conditions = [
        (df_clean['Impact_Temp'] >= df_clean['Impact_Mld']) & (df_clean['Impact_Temp'] >= df_clean['Impact_Light']),
        (df_clean['Impact_Mld'] >= df_clean['Impact_Temp']) & (df_clean['Impact_Mld'] >= df_clean['Impact_Light']),
        (df_clean['Impact_Light'] >= df_clean['Impact_Temp']) & (df_clean['Impact_Light'] >= df_clean['Impact_Mld'])
    ]
    choices = [1, 2, 3]
    df_clean['dominant_driver'] = np.select(conditions, choices, default=np.nan)
    df_clean = df_clean.dropna(subset=['dominant_driver'])
    
    # Forzar longitudes al formato centrado [-180, 180]
    lon_cl = (df_clean["lon"].to_numpy() + 180) % 360 - 180
    lat_cl = df_clean["lat"].to_numpy()
    val_cl = df_clean["dominant_driver"].to_numpy()
    
    # 2. CLAVE: Preparar los datos del hielo, pero purgando la grieta artificial de Greenwich
    df_ice_base = df.copy()
    lon_raw_fixed = (df_ice_base["lon"].to_numpy() + 180) % 360 - 180
    
    # Filtramos los NaNs artificiales que están entre -1° y 1.5° de longitud
    is_fake_nan = (lon_raw_fixed >= -1.0) & (lon_raw_fixed <= 1.5) & df_ice_base["Impact_Temp"].isna()
    df_ice_base = df_ice_base[~is_fake_nan]
    
    lon_raw = (df_ice_base["lon"].to_numpy() + 180) % 360 - 180
    lat_raw = df_ice_base["lat"].to_numpy()
    base_var = df_ice_base["Impact_Temp"].to_numpy()
    
    # 3. Acolchado Cíclico en los extremos del Pacífico (+/-150) para evitar cortes laterales
    pad_l = lon_cl < -150
    pad_r = lon_cl > 150
    lon_cl_p = np.concatenate([lon_cl, lon_cl[pad_l] + 360, lon_cl[pad_r] - 360])
    lat_cl_p = np.concatenate([lat_cl, lat_cl[pad_l], lat_cl[pad_r]])
    val_cl_p = np.concatenate([val_cl, val_cl[pad_l], val_cl[pad_r]])
    
    pad_raw_l = lon_raw < -150
    pad_raw_r = lon_raw > 150
    lon_raw_p = np.concatenate([lon_raw, lon_raw[pad_raw_l] + 360, lon_raw[pad_raw_r] - 360])
    lat_raw_p = np.concatenate([lat_raw, lat_raw[pad_raw_l], lat_raw[pad_raw_r]])
    base_var_p = np.concatenate([base_var, base_var[pad_raw_l], base_var[pad_raw_r]])
    
    # 4. Crear la grilla de alta resolución de -180 a 180 (Greenwich queda seguro en el centro)
    lon_grid = np.linspace(-180, 180, 1200)
    lat_grid = np.linspace(-90, 90, 1200)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    
    # 5. Interpolación Dual
    # 'nearest' para los drivers del océano
    grid_values = griddata((lon_cl_p, lat_cl_p), val_cl_p, (lon_mesh, lat_mesh), method="nearest")
    # 'linear' para el hielo (al no tener la grieta, unirá el Atlántico de forma continua)
    grid_mask = griddata((lon_raw_p, lat_raw_p), base_var_p, (lon_mesh, lat_mesh), method="linear")
    
    # Aplicamos el calado de hielo limpio (donde es NaN real, se borra)
    grid_values[np.isnan(grid_mask)] = np.nan
    
    # 6. Sello cíclico manual para cerrar herméticamente los bordes extremos del mapa
    lon_step = lon_grid[1] - lon_grid[0]
    lon_grid_cycl = np.append(lon_grid, lon_grid[-1] + lon_step)
    grid_values_cycl = np.hstack([grid_values, grid_values[:, :1]])
    lon_mesh_cycl, lat_mesh_cycl = np.meshgrid(lon_grid_cycl, lat_grid)
    
    # 7. Configurar la escena con Cartopy
    fig = plt.figure(figsize=(12, 6), dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    
    if zoom_spg:
        ax.set_extent([-80, 0, 35, 65], crs=ccrs.PlateCarree())
        
    # Paleta de 3 niveles
    colors = ['#E66101', '#5E3C99', '#4DAC26'] 
    cmap = mcolors.ListedColormap(colors)
    bounds = [0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Dibujar usando contornos vectoriales suaves continuos
    im = ax.contourf(
        lon_mesh_cycl, lat_mesh_cycl, grid_values_cycl, 
        transform=ccrs.PlateCarree(), 
        cmap=cmap, 
        norm=norm, 
        levels=bounds,
        zorder=1  
    )
    im.set_rasterized(True)
    
    # Colorbar
    cbar = plt.colorbar(im, orientation="vertical", shrink=0.5, pad=0.03, spacing='proportional')
    cbar.solids.set_rasterized(True) 
    cbar.solids.set_edgecolor("face")
    cbar.set_ticks([1.0, 2.0, 3.0])
    cbar.ax.set_yticklabels(['Thermal-Metabolic Driven', 'Physical Mixing Driven', 'Light Amplitude Driven'], fontsize=10)
    
    if zoom_spg:
        plt.title(f"SPG Phytoplankton Driver Control ({year})", fontsize=13, pad=10)
    
        output_dir = "figures"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/driver_dominance_spg_{year}.pdf", bbox_inches='tight', dpi=300)
    else:
        plt.title(f"Global Phytoplankton Driver Control ({year})", fontsize=13, pad=10)
    
        output_dir = "figures"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/driver_dominance_global_{year}.pdf", bbox_inches='tight', dpi=300)
        
    #plt.show()

def export_latex_results(cat_grid, lat_mesh, lon_mesh, labels, model_name="GFDL"):
    R = 6371.0  
    dlat = np.abs(np.radians(lat_mesh[1, 0] - lat_mesh[0, 0]))
    dlon = np.abs(np.radians(lon_mesh[0, 1] - lon_mesh[0, 0]))
    cell_areas = (R**2) * np.cos(np.radians(lat_mesh)) * dlat * dlon
    total_ocean_area = np.nansum(cell_areas[~np.isnan(cat_grid)])
    
    commands = [f"% --- {model_name} Automatic Variables ---"]
    table_rows = []
    
    for i, label in enumerate(labels):
        mask = (cat_grid == i)
        km2 = np.nansum(cell_areas[mask])
        m_km2 = km2 / 1e6 
        percentage = (km2 / total_ocean_area) * 100 if total_ocean_area > 0 else 0
        
        cmd_base = label.title().replace(" ", "").replace("-", "").replace("\n", "")
        
        cmd_area_name = f"{model_name}{cmd_base}Area"
        cmd_perc_name = f"{model_name}{cmd_base}Perc"
        
        commands.append(f"\\newcommand{{\\{cmd_area_name}}}{{{m_km2:.2f}}}")
        commands.append(f"\\newcommand{{\\{cmd_perc_name}}}{{{percentage:.1f}}}")
        
        table_rows.append({
            "Ecological Regime / Habitat": label.replace('\n', ' '),
            "Area ($10^6$ km$^2$)": f"\\{cmd_area_name}",
            "Ocean Coverage (%)": f"\\{cmd_perc_name}\\%"
        })

    output_path = "data/latex"
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "habitat_vars.tex"), "w") as f:
        f.write("\n".join(commands))

    with open(os.path.join(output_path, "habitat_table.tex"), "w") as f:
        f.write("\\begin{tabular}{p{7cm}cc}\n\\toprule\n")
        f.write("Ecological Regime / Habitat & Area ($10^6$ km$^2$) & Ocean Coverage (\\%) \\\\\n\\midrule\n")
        for row in table_rows:
            f.write(f"{row['Ecological Regime / Habitat']} & {row['Area ($10^6$ km$^2$)']} & {row['Ocean Coverage (%)']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}")
        
    print(f"✅ LaTeX outputs exported to {output_path}")
def export_hyperplane_stats(df, year="2024",model_name="GFDL", threshold=0.70):
    valid_r2 = df['R2'].dropna()
    total_records = len(valid_r2)
    
    n_above = (valid_r2 > threshold).sum()
    percentage_above = (n_above / total_records) * 100 if total_records > 0 else 0
    
    mean_r2 = valid_r2.mean()
    median_r2 = valid_r2.median()

    cmds = [
        f"% --- {model_name} Hyperplane Fit Statistics (Threshold: {threshold}) ---",
        f"\\newcommand{{\\{model_name}HyperThreshold}}{{{threshold:.2f}}}",
        f"\\newcommand{{\\{model_name}HyperTotalSamples}}{{{total_records:,}}}",
        f"\\newcommand{{\\{model_name}HyperAboveThresholdCount}}{{{n_above:,}}}",
        f"\\newcommand{{\\{model_name}HyperAboveThresholdPerc}}{{{percentage_above:.1f}}}",
        f"\\newcommand{{\\{model_name}HyperMeanRsq}}{{{mean_r2:.3f}}}",
        f"\\newcommand{{\\{model_name}HyperMedianRsq}}{{{median_r2:.3f}}}"
    ]
    output_path = "data/latex"
    os.makedirs(output_path, exist_ok=True)
    
    file_name = os.path.join(output_path, f"hyperplane_stats__{year}_{model_name.lower()}.tex")
    with open(file_name, "w") as f:
        f.write("\n".join(cmds))

    print(f"\n📊 Hyperplane Stats [{model_name}]")
    print("-" * 30)
    print(f"Total Valid Pixels : {total_records:,}")
    print(f"R² > {threshold}      : {n_above:,} ({percentage_above:.2f}%)")
    print(f"Mean R²           : {mean_r2:.4f}")
    print(f"File saved to      : {file_name}")

    return {
        "total": total_records,
        "count_above": n_above,
        "perc_above": percentage_above,
        "mean_r2": mean_r2
    }

def plot_global_shift_1d(df_2024, df_2100):
    """
    Generates a 1D density plot comparing the Global Ocean in 2024 vs 2100.
    Demonstrates the global shift towards thermal-metabolic dominance.
    """
    # Función interna para calcular el porcentaje térmico
    def get_thermal_pct(df):
        df_clean = clean_data(df).copy()
        df_clean = df_clean.dropna(subset=['Impact_Temp', 'Impact_Mld', 'Impact_Light'])
        
        total_impact = df_clean['Impact_Temp'] + df_clean['Impact_Mld'] + df_clean['Impact_Light']
        valid_mask = total_impact > 0
        df_clean = df_clean[valid_mask]
        total_impact = total_impact[valid_mask]
        
        return (df_clean['Impact_Temp'] / total_impact) * 100

    # 1. Calcular los porcentajes para ambos años
    thermal_2024 = get_thermal_pct(df_2024)
    thermal_2100 = get_thermal_pct(df_2100)
    
# 2. Configurar el gráfico
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    
    # Curva 2024 (Azul apagado para el baseline)
    sns.kdeplot(thermal_2024, fill=True, color="#4C72B0", alpha=0.4, linewidth=2, 
                label="Baseline (2024)", ax=ax)
    
    # Curva 2100 (Naranja para la proyección futura)
    sns.kdeplot(thermal_2100, fill=True, color="#E66101", alpha=0.5, linewidth=2, 
                label="Projection (2100)", ax=ax)
    
    # 3. Estética y ejes
    ax.set_xlim(0, 100)
    
    # Ejes estandarizados (sin negrita, tamaño 11 igual al label de tu mapa)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_xlabel("Thermal-Metabolic Driver Dominance (%)", fontsize=11)
    
    # Textos estandarizados (sin negrita, tamaño 10 igual a los ticks de tu mapa)
    ax.text(5, ax.get_ylim()[1] * 1.05, r"$\leftarrow$ Mixing Driven", color="#5E3C99", fontsize=10)
    ax.text(70, ax.get_ylim()[1] * 1.05, r"Thermal Driven $\rightarrow$", color="#E66101", fontsize=10)
    
    # Limpiar bordes innecesarios, PERO mantenemos el izquierdo ('left') activo
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Forzar notación científica en el eje Y
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Leyenda estandarizada
    ax.legend(frameon=False, fontsize=10, loc="upper center")
    
    # 4. Guardar
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/driver_density_1d_global_shift.pdf", bbox_inches='tight', dpi=300)
    #plt.show()
def plot_driver_shift_1d(df, year="2100"):
    """
    Genera un gráfico de densidad 1D superponiendo el océano global y el SPG.
    Demuestra cómo el SPG resiste la termalización global.
    """
    # 1. Limpieza inicial
    df_clean = clean_data(df).copy()
    df_clean = df_clean.dropna(subset=['Impact_Temp', 'Impact_Mld', 'Impact_Light'])
    
    # 2. Calcular el porcentaje térmico por píxel
    total_impact = df_clean['Impact_Temp'] + df_clean['Impact_Mld'] + df_clean['Impact_Light']
    valid_mask = total_impact > 0
    df_clean = df_clean[valid_mask]
    total_impact = total_impact[valid_mask]
    
    # Solo necesitamos el eje térmico, ya que (Mezcla = 100 - Térmico)
    df_clean['thermal_pct'] = (df_clean['Impact_Temp'] / total_impact) * 100
    
    # 3. Separar los datos en Global vs SPG
    # Corregir longitud a [-180, 180]
    lon_fixed = (df_clean['lon'] + 180) % 360 - 180
    lat = df_clean['lat']
    
    # Máscara del SPG (mismos límites de Cartopy)
    spg_mask = (lon_fixed >= -80) & (lon_fixed <= 0) & (lat >= 35) & (lat <= 65)
    
    # Extraer las series de datos
    global_thermal = df_clean['thermal_pct']
    spg_thermal = df_clean.loc[spg_mask, 'thermal_pct']
    

    # 2. Configurar el gráfico
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    
    # Curva 2024 (Azul apagado para el baseline)
    sns.kdeplot(global_thermal, fill=True, color="#E66101", alpha=0.4, linewidth=2, 
                label="Global Projection (2100)", ax=ax)
    
    # Curva 2100 (Naranja para la proyección futura)
    sns.kdeplot(spg_thermal, fill=True, color="#4C72B0", alpha=0.5, linewidth=2, 
                label="Subpolar Gyre (SPG)", ax=ax)

    # 3. Estética y ejes
    ax.set_xlim(0, 100)
    
    # Ejes estandarizados (sin negrita, tamaño 11 igual al label de tu mapa)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_xlabel("Thermal-Metabolic Driver Dominance (%)", fontsize=11)
    
    # Textos estandarizados (sin negrita, tamaño 10 igual a los ticks de tu mapa)
    ax.text(5, ax.get_ylim()[1] * 1.05, r"$\leftarrow$ Mixing Driven", color="#5E3C99", fontsize=10)
    ax.text(70, ax.get_ylim()[1] * 1.05, r"Thermal Driven $\rightarrow$", color="#E66101", fontsize=10)
    
    # Limpiar bordes innecesarios, PERO mantenemos el izquierdo ('left') activo
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Forzar notación científica en el eje Y
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Leyenda estandarizada
    ax.legend(frameon=False, fontsize=10, loc="upper center")
    
    # 6. Guardar
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/driver_density_1d_spg_{year}.pdf", bbox_inches='tight', dpi=300)
    #plt.show()


if __name__ == "__main__":


    df_2024_g = pd.read_parquet("/home/matiasneto/git/NPZ-T_Persistence/data/g_critic_2024_ssp585_GFDL-ESM4.parquet")

    plot_g_critic(df_2024_g)

    df_2024_h = pd.read_parquet("data1/hyperplane_2024_ssp585_GFDL-ESM4.parquet")

    df_2100_h = pd.read_parquet("data1/hyperplane_2100_ssp585_GFDL-ESM4.parquet")

    plot_driver_dominance(df_2024_h, year="2024", zoom_spg=False)

    plot_driver_dominance(df_2100_h, year="2100", zoom_spg=False)

    export_hyperplane_stats(df_2024_h, year="2024", model_name="GFDL", threshold=0.70)

    export_hyperplane_stats(df_2100_h, year="2100", model_name="GFDL", threshold=0.70)

    plot_global_shift_1d(df_2024_h, df_2100_h)

    plot_driver_shift_1d(df_2100_h, year="2100")






