
import pandas as pd
import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from scipy.interpolate import griddata
from npz_t_engine import give_rates, params_map

def fix_longitude(lon):
    lon_fixed = ((lon + 180) % 360) - 180
    return lon_fixed

def plot_growth_rate(n_steps=2000):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.direction": "in",
        "ytick.direction": "in"
    })

    temp_c = np.linspace(0, 42, n_steps)
    T_kelvin = (temp_c + 273.15)

    G_p = give_rates(T_kelvin, params_map['gp'])

    plt.figure(figsize=(7, 5), dpi=300)
    
    plt.fill_between(temp_c, G_p, color='teal', alpha=0.1)
    
    plt.plot(temp_c, G_p, color='#008080', lw=2.5, label=r'Metabolic Growth Rate $\mu_P(T)$')

    idx_opt = np.argmax(G_p)
    t_opt = temp_c[idx_opt]
    g_max = G_p[idx_opt]
    
    plt.axvline(t_opt, color='red', linestyle='--', alpha=0.5, lw=1)
    plt.scatter(t_opt, g_max, color='red', s=40, zorder=5, label=f'$T_{{opt}}$ ({t_opt:.1f} °C)')

    plt.text(t_opt - 10, g_max/2, 'Thermal\nActivation', horizontalalignment='center', alpha=0.6, fontsize=9)
    plt.text(t_opt + 4, g_max/2, 'Thermal\nInactivation', horizontalalignment='center', alpha=0.6, fontsize=9)

    plt.title("Phytoplankton Thermal Performance Curve (TPC)", pad=15)
    plt.xlabel("Temperature (°C)")
    plt.ylabel(r"Maximum Growth Rate ($\mu_p$) [day$^{-1}$]")
    
    plt.xlim(0, 42)
    plt.ylim(0, g_max * 1.1)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.legend(frameon=False, loc='upper left')
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    output_dir = "figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f"{output_dir}/mu_p_metabolic_curve.pdf", bbox_inches='tight')

def plot_g_critic(df,year="2024",model="GFDL-ESM4"):
    lats = df["lat"].to_numpy()
    lons = fix_longitude(df["lon"].to_numpy())
    g_1 = df["g_critic"].to_numpy()
    g_2 = df["g_critic"].to_numpy()

    lon_grid = np.linspace(lons.min(), lons.max(), 1000)
    lat_grid = np.linspace(lats.min(), lats.max(), 1000)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    g_interp = griddata((lons, lats), g_1, (lon_mesh, lat_mesh), method="linear")

    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")

    limit = 1.0 
    levels = np.linspace(0, limit, 41)

    im = ax.contourf(
        lon_mesh, lat_mesh, g_interp,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        levels=levels,
        extend='both',
        zorder=1    
    )

    im.set_rasterized(True)


    cbar = plt.colorbar(im, orientation="vertical", shrink=0.7, pad=0.05)
    cbar.set_label(fr"$\gamma^{{{year}}}$")
    cbar.solids.set_rasterized(True) 
    cbar.solids.set_edgecolor("face")

    plt.title(f"Critical Nutrient Requirement ({year},{model})")
    output_dir = "figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f"figures/g_critic_{year}_{model}.pdf", bbox_inches='tight', dpi=300)
    #plt.show()

def plot_g_critic_shift(df1, df2,year1="2024",year2="2100",model="GFDL-ESM4", zoom_spg=False):
    lats = df1["lat"].to_numpy()
    lons = fix_longitude(df1["lon"].to_numpy())
    g_1 = df1["g_critic"].to_numpy()
    g_2 = df2["g_critic"].to_numpy()

    lon_grid = np.linspace(lons.min(), lons.max(), 1000)
    lat_grid = np.linspace(lats.min(), lats.max(), 1000)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    g_interp = griddata((lons, lats), g_2-g_1, (lon_mesh, lat_mesh), method="linear")

    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=11)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    if zoom_spg==True:
        ax.set_extent([-80, 0, 35, 65], crs=ccrs.PlateCarree())

    limit = 0.5 
    levels = np.linspace(-limit, limit, 41)

    im = ax.contourf(
        lon_mesh, lat_mesh, g_interp,
        transform=ccrs.PlateCarree(),
        cmap="PRGn_r",
        levels=levels,
        extend='both',
        zorder=1
    )
    im.set_rasterized(True)
    cbar = plt.colorbar(im, orientation="vertical", shrink=0.7, pad=0.05)
    cbar.set_label(fr"$\Delta \gamma=\gamma^{{{year1}}}-\gamma^{{{year2}}}$")
    cbar.solids.set_rasterized(True)
    cbar.solids.set_edgecolor("face")

    if zoom_spg==True:
        plt.title(f"Shift in Critical Nutrient Requirement SPG ({year1}--{year2},{model})")
        output_dir = "figures"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(f"figures/g_critic_shift_spg_{model}.pdf", bbox_inches='tight', dpi=300)
        #plt.show()
    else:
        plt.title(f"Shift in Critical Nutrient Requirement ({year1}--{year2},{model})")
        output_dir = "figures"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(f"figures/g_critic_shift_{model}.pdf", bbox_inches='tight', dpi=300)
        #plt.show()

def plot_biogeo_shift(df1,df2,year1="2024",year2="2100",model="GFDL-ESM4"):
    lats = df1["lat"].to_numpy()
    lons = fix_longitude(df1["lon"].to_numpy())
    g24_raw = df1["g_critic"].to_numpy()
    g21_raw = df2["g_critic"].to_numpy()


    lon_grid = np.linspace(lons.min(), lons.max(), 1000) 
    lat_grid = np.linspace(lats.min(), lats.max(), 1000)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)


    g24_grid = griddata((lons, lats), g24_raw, (lon_mesh, lat_mesh), method='linear')
    g21_grid = griddata((lons, lats), g21_raw, (lon_mesh, lat_mesh), method='linear')

    cat_grid = np.full_like(g24_grid, np.nan)

    is_num_24 = ~np.isnan(g24_grid)
    is_num_100 = ~np.isnan(g21_grid)
    v24 = g24_grid < 1.0
    v21 = g21_grid < 1.0

    cat_grid = np.where(is_num_24 & is_num_100 & v24 & v21, 0, cat_grid)

    cat_grid = np.where(is_num_24 & is_num_100 & ~v24 & ~v21, 1, cat_grid)

    cat_grid = np.where(is_num_24 & is_num_100 & ~v24 & v21, 2, cat_grid)

    cat_grid = np.where(is_num_24 & is_num_100 & v24 & ~v21, 3, cat_grid)

    cat_grid = np.where(~is_num_24 & is_num_100 & v21, 4, cat_grid)

    cat_grid = np.where(~is_num_24 & is_num_100 & ~v21, 5, cat_grid)

    ocean_colors = [
        "#AEDCF8", 
        '#5D6D7E', 
        "#81DAA7", 
        '#F1948A', 
        '#229954', 
        '#B03A2E'  
]
    cmap_cat = mcolors.ListedColormap(ocean_colors)
    bounds = [0, 1, 2, 3, 4, 5, 6]
    norm_cat = mcolors.BoundaryNorm(bounds, cmap_cat.N)

    fig = plt.figure(figsize=(12, 5), dpi=300) 
    ax = plt.axes(projection=ccrs.PlateCarree())


    ax.add_feature(cfeature.LAND, facecolor="#2C3E50", zorder=2) 
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, color='#444444', zorder=3)

    im = ax.pcolormesh(
        lon_mesh, lat_mesh, cat_grid,
        transform=ccrs.PlateCarree(),
        cmap=cmap_cat,
        norm=norm_cat,
        shading='nearest',
        alpha=0.9,
        zorder=1
    )
    im.set_rasterized(True)

    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.2, color='gray')

    cbar = plt.colorbar(
        im, 
        ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 
        orientation="horizontal", 
        shrink=0.7, 
        pad=0.1, 
        aspect=50   
    )

    labels = [
        'Stable\nViability', 
        'Stable\nRestriction', 
        'Habitat\nExpansion', 
        'Habitat\nContraction',
        'Ice-Free\nViability', 
        'Ice-Free\nRestriction'
    ]

    cbar.ax.set_xticklabels(labels, fontsize=8, ha='center')
    cbar.outline.set_visible(False)
    cbar.solids.set_rasterized(True) 
    cbar.solids.set_edgecolor("face")

    plt.title(f"Biogeographical Shifts in Critical Threshold ({year1}–-{year2},{model})")
    output_dir = "figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f"figures/biogeo_shifts_{model}.pdf", bbox_inches='tight', dpi=300)
    #plt.show()
    return labels, cat_grid, lat_mesh, lon_mesh

def export_latex_results(cat_grid, lat_mesh, lon_mesh, labels, model_name="GFDL"):
    R = 6371.0  
    dlat = np.abs(np.radians(lat_mesh[1, 0] - lat_mesh[0, 0]))
    dlon = np.abs(np.radians(lon_mesh[0, 1] - lon_mesh[0, 0]))
    
    cell_areas = (R**2) * np.cos(np.radians(lat_mesh)) * dlat * dlon
    total_ocean_area = np.nansum(cell_areas[~np.isnan(cat_grid)])
    
    commands = [f"% --- {model_name} Automatic Variables ---"]
    table_rows = []
    
    def clean_cmd_name(name):
        clean = "".join(filter(str.isalpha, name.title()))
        return clean

    for i, label in enumerate(labels):
        mask = (cat_grid == i)
        km2 = np.nansum(cell_areas[mask])
        m_km2 = km2 / 1e6 
        percentage = (km2 / total_ocean_area) * 100 if total_ocean_area > 0 else 0
        
        cmd_base = clean_cmd_name(label)

        cmd_area_name = f"{model_name}{cmd_base}Area"
        cmd_perc_name = f"{model_name}{cmd_base}Perc"
        
        commands.append(f"\\newcommand{{\\{cmd_area_name}}}{{{m_km2:.2f}}}")
        commands.append(f"\\newcommand{{\\{cmd_perc_name}}}{{{percentage:.1f}}}")
        
        table_rows.append({
            "Ecological Regime / Habitat": label.replace('\n', ' '),
            "Area ($10^6$ km$^2$)": f"\\{cmd_area_name}",
            "Ocean Coverage (%)": f"\\{cmd_perc_name}\\%"
        })

    df_stats = pd.DataFrame(table_rows)
    table_tex = df_stats.to_latex(
        index=False,
        escape=False,
        caption=f"Global distribution of phytoplankton regimes and habitat statistics derived from {model_name} projections.",
        label=f"tab:habitat_stats_{model_name.lower()}",
        column_format="p{7cm}cc",
        position="t"
    )

    output_path = "data/latex"
    os.makedirs(output_path, exist_ok=True)
    
    vars_file = os.path.join(output_path, "habitat_vars.tex")
    with open(vars_file, "w") as f:
        f.write("\n".join(commands))
        
    table_file = os.path.join(output_path, "habitat_table.tex")
    with open(table_file, "w") as f:
        f.write(table_tex)
        
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

def plot_thermal_index(df,year="2024",zoom_spg=False):
    metric_name = 'sens'
    lats = df["lat"].to_numpy()
    lons = fix_longitude(df["lon"].to_numpy())
    values = df[metric_name].to_numpy()

    lon_grid = np.linspace(lons.min(), lons.max(), 1000)
    lat_grid = np.linspace(lats.min(), lats.max(), 1000)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    grid_values = griddata((lons, lats), values, (lon_mesh, lat_mesh), method="linear")

    fig = plt.figure(figsize=(12, 6), dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor='#F0F0F0', zorder=2) 
    ax.add_feature(cfeature.COASTLINE, edgecolor='#404040', linewidth=0.6, zorder=3)
    ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
    if zoom_spg==True:
        ax.set_extent([-80, 0, 35, 65], crs=ccrs.PlateCarree())

    v_min, v_max = np.nanmin(grid_values), np.nanmax(grid_values)

    if v_min < 0 and v_max > 0:
        norm = mcolors.TwoSlopeNorm(vmin=v_min, vcenter=0, vmax=v_max)
        cmap = "RdBu_r"
    elif v_min >= 0:
        norm = mcolors.Normalize(vmin=v_min, vmax=v_max)
        cmap = "RdBu_r" 
    else:
        norm = mcolors.Normalize(vmin=v_min, vmax=v_max)
        cmap = "Blues_r"

    im = ax.contourf(
        lon_mesh, lat_mesh, grid_values, 
        transform=ccrs.PlateCarree(), 
        cmap=cmap, 
        levels=100, 
        norm=norm, 
        extend='both',
        zorder=1  
    )
    im.set_rasterized(True)

    cbar = plt.colorbar(im, orientation="vertical", shrink=0.6, pad=0.03)
    cbar.solids.set_rasterized(True) 
    cbar.solids.set_edgecolor("face")

    cbar.set_label(fr"$\mathcal{{D}}^{{{year}}}_T(x)$", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    if zoom_spg==False:
        plt.title(f"Global Thermal Dominance Index ({year})")

        output_dir = "figures"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(f"figures/thermal_dominance_temp_{year}.pdf", bbox_inches='tight', dpi=300)
        #plt.show()
    else:
        plt.title(f"SPG Thermal Dominance Index ({year})")
        output_dir = "figures"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(f"figures/spg_thermal_dominance_temp_{year}.pdf", bbox_inches='tight', dpi=300)
        #plt.show()


if __name__ == "__main__":

    plot_growth_rate()

    df_2024_g = pd.read_parquet("/home/matiasneto/git/NPZ-T_Persistence/data/g_critic_2024_ssp585_GFDL-ESM4.parquet")

    plot_g_critic(df_2024_g)

    df_2100_g = pd.read_parquet("/home/matiasneto/git/NPZ-T_Persistence/data/g_critic_2100_ssp585_GFDL-ESM4.parquet")

    plot_g_critic_shift(df_2024_g,df_2100_g)

    plot_g_critic_shift(df_2024_g,df_2100_g,zoom_spg=True)

    labels, cat_grid, lat_mesh, lon_mesh = plot_biogeo_shift(df_2024_g,df_2100_g)

    export_latex_results(cat_grid,lat_mesh,lon_mesh,labels)

    df_hyper_2024_g = pd.read_parquet("/home/matiasneto/git/NPZ-T_Persistence/data/hyperplane_2024_ssp585_GFDL-ESM4.parquet")

    df_hyper_2100_g = pd.read_parquet("/home/matiasneto/git/NPZ-T_Persistence/data/hyperplane_2100_ssp585_GFDL-ESM4.parquet")

    export_hyperplane_stats(df_hyper_2024_g, threshold=0.8)

    export_hyperplane_stats(df_hyper_2100_g, threshold=0.8, year="2100")

    plot_thermal_index(df_hyper_2024_g)

    plot_thermal_index(df_hyper_2100_g,year="2100")

    plot_thermal_index(df_hyper_2100_g,year="2100",zoom_spg=True)







