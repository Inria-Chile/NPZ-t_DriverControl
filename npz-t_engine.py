import numpy as np
import swifter 
from scipy.optimize import curve_fit

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


def Constants(m):
  C,H =-14418.93, 4213943.43
  ln=np.log(m)
  S0=(ln-np.log(kB/h_d)-np.log(T0)+H/(R*T0))*R
  return C, H, S0, ln

def get_params_arroyo(val_base):
    C, H, S0, ln = Constants(val_base)
    const_part = np.log(kB / h_d) + (S0 / R) - (C / R) * np.log(T0)
    return {'C': C, 'H': H, 'const_part': const_part}

params_map = {
    'gp': get_params_arroyo(gp),
    'mp': get_params_arroyo(mp)
}


def give_rates(T_kelvin, p):
    return np.exp(((p['C'] / R) + 1) * np.log(T_kelvin) + p['const_part'] - (p['H'] / (R * T_kelvin)))


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

def calcular_tiempo(fecha):
    dias_del_mes = fecha.days_in_month
    dia_actual = fecha.day
    mes = fecha.month

    tiempo = mes - 1 + dia_actual / dias_del_mes

    return tiempo

def fix_longitude(lon):
    lon_fixed = ((lon + 180) % 360) - 180
    return lon_fixed

def compute_g_critic(row, n_steps=2000, params_calc=params_map,i_0=40):
    temp_vec = row["temperature_vector"]
    depth_vec = row["mld_vector"]
    lat = row["lat"]
    
    if temp_vec[0] is None or depth_vec[0] is None or len(temp_vec) == 0:
        return np.nan

    H0_mid, s0 = give_h_smas(n_steps, depth_vec)
    
    dH = (1 / (K_d * H0_mid)) * (1 - np.exp(-K_d * H0_mid))

    temp_fit_vec = fit_temp_vector(temp_vec, n_points=n_steps)
    T_kelvin_mid = (temp_fit_vec + 273.15)[1:-1]


    G_p_mid = give_rates(T_kelvin_mid, params_calc['gp'])
    M_p_mid = give_rates(T_kelvin_mid, params_calc['mp'])
    
    t_mid = np.linspace(0, 365, n_steps)[1:-1]
    I_mid = get_light_forcing(lat, t_mid) 
    
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