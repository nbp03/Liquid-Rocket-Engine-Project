import numpy as np
import pandas as pd
from scipy.constants import h, c, k
from radis import SpectrumFactory
import warnings
import logging
import os
import gc

warnings.filterwarnings("ignore")
os.environ["RADIS_IGNORE_WARNINGS"] = "1"
logging.getLogger("radis").setLevel(logging.CRITICAL)

try:
    integrate_area = np.trapezoid
except AttributeError:
    integrate_area = np.trapz

# ==========================================
# 1. PARAMETERS
# ==========================================
X_H2O, X_CO2, X_CO = 0.50, 0.34, 0.08
T_list = np.linspace(1000, 3500, 10)
L_list = np.logspace(-1, 2, 10)
P_atm = 5.0
C2 = (h * c * 100.0) / k 
PI4_15 = (np.pi**4) / 15.0

nu_master = np.linspace(1.0, 8000.0, 200000)

# Dictionaries to hold our tiny 1.6 MB math arrays for each temperature
k_h2o_dict = {}
k_co2_dict = {}
k_co_dict  = {}

print("Starting Optimized Molecule-First Integration...")

# =========================================================
# PHASE 1: WATER (H2O)
# =========================================================
print("\n--- PHASE 1: H2O ---")
print("Loading massive H2O database into RAM (Takes 3-5 mins, but ONLY ONCE!)...")

# save_memory is FALSE so we can keep the factory alive for all 10 temps!
sf_h2o = SpectrumFactory(wavenum_min=1.0, wavenum_max=8000.0, pressure=P_atm, wstep=0.05, 
                         isotope='1', molecule='H2O', save_memory=False, cutoff=1e-27, truncation=50)
sf_h2o.load_databank('HITEMP-H2O')

for T in T_list:
    print(f"  -> Calculating H2O at {T:.1f} K...")
    s_h2o = sf_h2o.eq_spectrum(Tgas=T, mole_fraction=X_H2O, path_length=1.0)
    nu_h2o = s_h2o.get('abscoeff')[0]
    k_raw = s_h2o.get('abscoeff')[1]
    # Stretch array to fit master grid and save it in the dictionary
    k_h2o_dict[T] = np.interp(nu_master, nu_h2o, k_raw)

print("H2O Complete! Nuking massive H2O database from RAM...")
del sf_h2o, s_h2o, nu_h2o, k_raw
gc.collect()

# =========================================================
# PHASE 2: CARBON DIOXIDE (CO2)
# =========================================================
print("\n--- PHASE 2: CO2 ---")
print("Loading CO2 database...")
sf_co2 = SpectrumFactory(wavenum_min=1.0, wavenum_max=8000.0, pressure=P_atm, wstep=0.05, 
                         isotope='1', molecule='CO2', save_memory=False, cutoff=1e-27, truncation=50)
sf_co2.load_databank('HITEMP-CO2')

for T in T_list:
    print(f"  -> Calculating CO2 at {T:.1f} K...")
    s_co2 = sf_co2.eq_spectrum(Tgas=T, mole_fraction=X_CO2, path_length=1.0)
    nu_co2 = s_co2.get('abscoeff')[0]
    k_raw = s_co2.get('abscoeff')[1]
    k_co2_dict[T] = np.interp(nu_master, nu_co2, k_raw)

print("CO2 Complete! Nuking CO2 database from RAM...")
del sf_co2, s_co2, nu_co2, k_raw
gc.collect()

# =========================================================
# PHASE 3: CARBON MONOXIDE (CO)
# =========================================================
print("\n--- PHASE 3: CO ---")
print("Loading CO database...")
sf_co = SpectrumFactory(wavenum_min=1.0, wavenum_max=8000.0, pressure=P_atm, wstep=0.05, 
                        isotope='1', molecule='CO', save_memory=False, cutoff=1e-27, truncation=50)
sf_co.load_databank('HITEMP-CO')

for T in T_list:
    print(f"  -> Calculating CO at {T:.1f} K...")
    s_co = sf_co.eq_spectrum(Tgas=T, mole_fraction=X_CO, path_length=1.0)
    nu_co = s_co.get('abscoeff')[0]
    k_raw = s_co.get('abscoeff')[1]
    k_co_dict[T] = np.interp(nu_master, nu_co, k_raw)

print("CO Complete! Nuking CO database from RAM...")
del sf_co, s_co, nu_co, k_raw
gc.collect()

# =========================================================
# PHASE 4: MIXING & INTEGRATION (Instantaneous)
# =========================================================
print("\n--- PHASE 4: MIXING AND INTEGRATING ---")
results = []

for T in T_list:
    # Add the 3 tiny stored arrays together for this specific temperature
    k_mix = k_h2o_dict[T] + k_co2_dict[T] + k_co_dict[T]
    
    planck_dist = (nu_master**3) / (np.exp(C2 * nu_master / T) - 1.0)
    denominator_total_bb = ((T / C2)**4) * PI4_15
    
    for L in L_list:
        tau_mix = np.exp(-k_mix * L)
        emissivity_nu = 1.0 - tau_mix
        numerator_active_bb = integrate_area(emissivity_nu * planck_dist, nu_master)
        total_emissivity = numerator_active_bb / denominator_total_bb
        
        results.append({
            'Temperature_K': T,
            'Path_Length_cm': L,
            'Pressure_atm': P_atm,
            'PL_atm_cm': P_atm * L,
            'Broadband_Emissivity': total_emissivity
        })

# ==========================================
# 5. EXPORT TO CSV
# ==========================================
df = pd.DataFrame(results)
output_filename = 'gas_emissivity_grid.csv'
df.to_csv(output_filename, index=False)
print(f"\nSUCCESS! Emissivity table perfectly calculated and saved to {output_filename}")