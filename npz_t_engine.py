import numpy as np
import swifter
import pandas as pd
from sklearn.linear_model import LinearRegression

# Thermodynamic Constants
T0 = 293.15
kB = 1.380649 * 10 ** (-23)
h = 6.62607015 * 10 ** (-34)
h_d = h * 86400
R = 8.31446261815324
K_d = 0.05

#Biological Parameters at 20°C
gp=0.8
mp=0.024

# NPZ-T Engine

def mldmodel(t, ml_max):
    ml_min, sl, N, M = 25, 25, 80, 80
    
    t1 = t * (30.41)
    tm = (t1 - 270) % 365

    mask1 = tm < (M + 0.0001)
    mask2 = tm > M

    x1 = np.pi * (tm / M * (2 - tm / M)) * mask1
    x2 = (np.pi + (np.pi / (365 - sl)**3) * (tm - sl)**3) * mask2

    return (ml_min + ml_max) / 2 + (ml_max - ml_min) / 2 * np.cos(x1 + x2)

def give_h_smas(n_steps, depth_vec):
    depth_vec = np.atleast_1d(np.asarray(depth_vec, dtype=float))
    
    local_max = np.max(depth_vec)
    days_12 = np.linspace(0, 12, len(depth_vec))
    
    H_vec_model = mldmodel(days_12, local_max)
    shift = days_12[np.argmax(depth_vec)] - days_12[np.argmax(H_vec_model)]

    days_high = np.linspace(0, 12, n_steps)
    H0 = mldmodel(days_high - shift, local_max)

    t_total = np.linspace(0, 365, n_steps)
    DH0 = (H0[2:] - H0[:-2]) / (t_total[2:] - t_total[:-2])
    
    s0 = np.maximum(0.0, DH0 / H0[1:-1]) 
    
    H0_mid = H0[1:-1]
    
    return H0_mid, s0

def fit_temp_vector(temp_vector, n_points=2000):
    t_months = np.arange(1, 13) * 30.41 
    y = np.array(temp_vector)
    
    b_fijo = 2 * np.pi / 365
    X1 = np.cos(b_fijo * t_months)
    X2 = np.sin(b_fijo * t_months)
    
    c = np.mean(y)
    
    y_centrado = y - c
    A = (2/12) * np.sum(y_centrado * X1)
    B = (2/12) * np.sum(y_centrado * X2)

    t_fit = np.linspace(0, 365, n_points)
    y_fit = A * np.cos(b_fijo * t_fit) + B * np.sin(b_fijo * t_fit) + c
    
    return y_fit


def get_params_arroyo(val_base):
    C = -14418.93
    H = 4213943.43
    
    return {'Y_0': val_base, 'C': C, 'H': H}

params_map = {
    'gp': get_params_arroyo(gp),
    'mp': get_params_arroyo(mp)
}

def give_rates(T_kelvin, p):
    Y_0 = p['Y_0']
    C = p['C']
    H = p['H']

    term_C = (C / R) * np.log(T_kelvin / T0)
    term_H = (H / R) * ((1.0 / T0) - (1.0 / T_kelvin))
    
    rate = Y_0 * np.exp(term_C + term_H)
    
    return rate


def calculate_theoretical_insolation(lat_deg, day_of_year):
    phi = np.radians(lat_deg)
    
    delta = 0.409 * np.sin((2 * np.pi / 365) * day_of_year - 1.39)
    
    argument = -np.tan(phi) * np.tan(delta)
    
    argument = np.clip(argument, -1.0, 1.0)
    omega_s = np.arccos(argument)
    
    dr = 1 + 0.033 * np.cos((2 * np.pi / 365) * day_of_year)
    
    S_solar = 1360.0 

    Ra = (S_solar / np.pi) * dr * (
        omega_s * np.sin(phi) * np.sin(delta) +
        np.cos(phi) * np.cos(delta) * np.sin(omega_s)
    )
    
    return np.maximum(Ra, 0.0)


lat_ref = 47.0     
target_mean = 20.0 
days = np.arange(1, 366)

I_raw_ref = calculate_theoretical_insolation(lat_ref, days)
mean_raw_ref = np.mean(I_raw_ref)

scaling_factor = target_mean / mean_raw_ref


def get_light_forcing(lat, t):

    day = (t % 365) + 1 
    
    raw_light = calculate_theoretical_insolation(lat, day)
    return raw_light * scaling_factor

def fit_light_vector(light_vec, n_steps=2000, scaling=1.0):
    month_days = np.linspace(15, 345, 12)
    omega = 2 * np.pi / 365
    
    X = np.vstack([
        np.ones_like(month_days), 
        np.cos(omega * month_days), 
        np.sin(omega * month_days)
    ]).T
    
    beta, _, _, _ = np.linalg.lstsq(X, light_vec, rcond=None)
    
    I_mean = beta[0]

    I_amp = np.sqrt(beta[1]**2 + beta[2]**2)
    phase = np.arctan2(beta[1], beta[2])
    t_mid = np.linspace(0, 365, n_steps)[1:-1]

    I_amp_realized = I_amp * scaling
    
    omega = 2 * np.pi / 365
    I_mid_raw = I_mean + I_amp_realized * np.sin(omega * t_mid + phase)
    
    I_mid = np.maximum(I_mid_raw, 0.0)
    
    return I_mid, I_amp_realized, I_mean, omega, phase

def compute_g_critic(row, n_steps=2000, params_calc=params_map,i_0=40):
    temp_vec = row["temperature_vector"]
    depth_vec = row["mld_vector"]
    light_vec = row["light_vector"]
    
    if temp_vec[0] is None or depth_vec[0] is None or len(temp_vec) == 0:
        return np.nan

    H0_mid, s0 = give_h_smas(n_steps, depth_vec)
    
    dH = (1 / (K_d * H0_mid)) * (1 - np.exp(-K_d * H0_mid))

    temp_fit_vec = fit_temp_vector(temp_vec, n_points=n_steps)
    T_kelvin_mid = (temp_fit_vec + 273.15)[1:-1]


    G_p_mid = give_rates(T_kelvin_mid, params_calc['gp'])
    M_p_mid = give_rates(T_kelvin_mid, params_calc['mp'])
    
    t_mid = np.linspace(0, 365, n_steps)[1:-1]
    I_mid, I_amp_realized, I_mean, omega, phase = fit_light_vector(light_vec)
    
    mu = G_p_mid * (I_mid / (I_mid + i_0)) * dH
    l = M_p_mid + s0
    
    G_total = np.trapezoid(mu, t_mid)
    L_total = np.trapezoid(l, t_mid)
    
    return L_total / G_total 
def apply_g_critic(df):
    valid_df = df[
        df.apply(
            lambda row: not np.all(np.isnan(row["temperature_vector"])) and not np.all(np.isnan(row["mld_vector"])), axis=1
        )
    ]
    df["g_critic"] = valid_df.swifter.apply(compute_g_critic, axis=1)
    return df
def hyperplane_balanced(n_steps, temp_vec, depth_vec, light_vec, params_calc=params_map):
    mld_max = np.max(depth_vec)
    idx_max = np.argmax(depth_vec)
    
    ML_vals = np.linspace(max(10, mld_max - 50), mld_max + 50, 5)
    DELTA_vals = np.linspace(-3.0, 3.0, 5)
    
    ML_grid, DELTA_grid = np.meshgrid(ML_vals, DELTA_vals)
    ML_list = ML_grid.flatten()
    DELTA_list = DELTA_grid.flatten()
    
    results = np.zeros_like(ML_list)
    temp_fit_base = fit_temp_vector(temp_vec, n_points=n_steps)
    
    t_mid = np.linspace(0, 365, n_steps)[1:-1]
    I_mid, I_amp_realized, I_mean, omega, phase = fit_light_vector(light_vec)
    
    for i in range(len(ML_list)):
        d_vec_temp = depth_vec.copy()
        d_vec_temp[idx_max] = ML_list[i]
        H0_mid, s0 = give_h_smas(n_steps, d_vec_temp)
        dH = (1 / (K_d * H0_mid)) * (1 - np.exp(-K_d * H0_mid))
        
        T_kelvin_mid = (temp_fit_base + 273.15 + DELTA_list[i])[1:-1]
        G_p = give_rates(T_kelvin_mid, params_calc['gp'])
        M_p = give_rates(T_kelvin_mid, params_calc['mp'])
        
        mu = G_p * (I_mid / (I_mid + 40)) * dH
        G_total = np.trapezoid(mu, t_mid)
        L_total = np.trapezoid(M_p+s0, t_mid)
    
        results[i] = L_total / G_total 

    X = np.column_stack((DELTA_list, ML_list))
    reg = LinearRegression().fit(X, results)
    
    return reg.coef_[0], reg.coef_[1], reg.intercept_, reg.score(X, results)

def compute_hyperplane(row, n_steps=2000, params_calc=None):

    lat = row["lat"]
    temp_vec = np.array(row["temperature_vector"], dtype=float)
    depth_vec = np.array(row["mld_vector"], dtype=float)

    if temp_vec is None or depth_vec is None or len(temp_vec) == 0 or np.isnan(temp_vec).any():
        return pd.Series({
            'S_temp': np.nan, 
            'S_mld': np.nan, 
            'Intercept': np.nan, 
            'R2': np.nan
        })


    a, b, c, r2 = hyperplane_balanced(n_steps, temp_vec, depth_vec, lat, params_calc)
        
    return pd.Series({
        'S_temp': a,      
        'S_mld': b,      
        'Intercept': c,  
        'R2': r2          
    })
def apply_hyperplane(df, n_steps=2000, params_calc=params_map):
    mask = df.apply(
        lambda row: (row["temperature_vector"] is not None) and 
                    (row["mld_vector"] is not None) and 
                    (len(row["temperature_vector"]) > 0), 
        axis=1
    )
    

    hyperplane_results = df[mask].swifter.apply(
        compute_hyperplane, 
        axis=1, 
        n_steps=n_steps, 
        params_calc=params_calc
    )

    df = pd.concat([df, hyperplane_results], axis=1)
    
    return df

def hyperplane_3d(n_steps, temp_vec, depth_vec, light_vec, params_calc=params_map):
    # 1. Parámetros base de la Capa de Mezcla (MLD)
    mld_max = np.max(depth_vec)
    idx_max = np.argmax(depth_vec)
    
    # 2. Extraer los parámetros analíticos de la luz real CMIP6
    I_mid, I_amp_realized, I_mean, omega, phase = fit_light_vector(light_vec)
    
    # 3. Definir los tres espacios de perturbación local
    ML_vals = np.linspace(max(10, mld_max - 50), mld_max + 50, 5)
    DELTA_vals = np.linspace(-3.0, 3.0, 5)
    # Perturbamos la amplitud de la luz de forma absoluta (ej: +/- 15% de su rango local)
    AMP_deltas = np.linspace(-0.15 * (I_amp_realized + 1.0), 0.15 * (I_amp_realized + 1.0), 5)
    
    # 4. Crear la malla tridimensional (5x5x5 = 125 combinaciones analíticas)
    DELTA_grid, ML_grid, AMP_grid = np.meshgrid(DELTA_vals, ML_vals, AMP_deltas, indexing='ij')
    
    DELTA_list = DELTA_grid.flatten()
    ML_list = ML_grid.flatten()
    AMP_list = AMP_grid.flatten()
    
    results = np.zeros_like(ML_list)
    temp_fit_base = fit_temp_vector(temp_vec, n_points=n_steps)
    t_mid = np.linspace(0, 365, n_steps)[1:-1]
    omega = 2 * np.pi / 365
    
    # 5. Loop de evaluación del resolvedor escalar de Floquet
    for i in range(len(ML_list)):
        # Aplicar perturbación a la MLD
        d_vec_temp = depth_vec.copy()
        d_vec_temp[idx_max] = ML_list[i]
        H0_mid, s0 = give_h_smas(n_steps, d_vec_temp)
        dH = (1 / (K_d * H0_mid)) * (1 - np.exp(-K_d * H0_mid))
        
        # Aplicar perturbación a la Temperatura
        T_kelvin_mid = (temp_fit_base + 273.15 + DELTA_list[i])[1:-1]
        G_p = give_rates(T_kelvin_mid, params_calc['gp'])
        M_p = give_rates(T_kelvin_mid, params_calc['mp'])
        
        # Aplicar perturbación analítica a la Amplitud de la Luz
        I_amp_perturbed = I_amp_realized + AMP_list[i]  # <-- Aquí ocurre la magia de la derivada
        I_mid_raw = I_mean + I_amp_perturbed * np.sin(omega * t_mid + phase)
        I_mid = np.maximum(I_mid_raw, 0.0)
        
        # Integración numérica del umbral de viabilidad
        mu = G_p * (I_mid / (I_mid + 40)) * dH
        G_total = np.trapezoid(mu, t_mid)
        L_total = np.trapezoid(M_p + s0, t_mid)
    
        results[i] = L_total / G_total 

    # 6. Regresión Lineal Múltiple en 3D
    X = np.column_stack((DELTA_list, ML_list, AMP_list))
    reg = LinearRegression().fit(X, results)
    
    # Retorna las 3 derivadas parciales locales, el intercepto y la bondad de ajuste (R2)
    return reg.coef_[0], reg.coef_[1], reg.coef_[2], reg.intercept_, reg.score(X, results)


def compute_hyperplane_3d(row, n_steps=2000, params_calc=None):
    temp_vec = np.array(row["temperature_vector"], dtype=float)
    depth_vec = np.array(row["mld_vector"], dtype=float)
    light_vec = np.array(row["light_vector"], dtype=float)  # <-- Extraído correctamente

    # Control de seguridad para celdas terrestres
    if (temp_vec is None or depth_vec is None or light_vec is None or 
        len(temp_vec) == 0 or np.isnan(temp_vec).any() or np.isnan(light_vec).any()):
        return pd.Series({
            'S_temp': np.nan, 
            'S_mld': np.nan, 
            'S_light': np.nan, 
            'Intercept': np.nan, 
            'R2': np.nan
        })

    # CORRECCIÓN: Pasamos el vector de luz en lugar de la latitud
    a, b, d, c, r2 = hyperplane_3d(n_steps, temp_vec, depth_vec, light_vec, params_calc)
        
    return pd.Series({
        'S_temp': a,       # d(g_crit)/dT  (Tasa de cambio ante el promedio térmico)
        'S_mld': b,        # d(g_crit)/dMLD (Tasa de cambio ante el máximo invernal)
        'S_light': d,      # d(g_crit)/dI_amp (¡Tu nueva derivada ante la amplitud estacional!)
        'Intercept': c,  
        'R2': r2          
    })


def apply_hyperplane_3d(df, n_steps=2000, params_calc=params_map):
    mask = df.apply(
        lambda row: (row["temperature_vector"] is not None) and 
                    (row["mld_vector"] is not None) and 
                    (row["light_vector"] is not None) and 
                    (len(row["temperature_vector"]) > 0), 
        axis=1
    )
    
    print("🚀 Computing 3D Hyperplane Sensitivities globally...")
    hyperplane_results = df[mask].swifter.apply(
        compute_hyperplane_3d, 
        axis=1, 
        n_steps=n_steps, 
        params_calc=params_calc
    )

    df = pd.concat([df, hyperplane_results], axis=1)
    return df