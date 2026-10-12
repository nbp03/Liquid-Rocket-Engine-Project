# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.linalg import solve_banded
import cantera as ct
from rocketcea.cea_obj_w_units import CEA_Obj
from rocketcea.cea_obj import add_new_fuel
import pandas as pd
from scipy.interpolate import interp1d, LinearNDInterpolator
import sys

def tand(deg): return np.tan(np.radians(deg))
def sind(deg): return np.sin(np.radians(deg))
def cosd(deg): return np.cos(np.radians(deg))
def acotd(x): return np.degrees(np.arctan(1/x))

def solve_mach_from_area(AR, gamma, is_subsonic):
    """Solves the isentropic area-Mach relation using a root-finding algorithm."""
    if AR < 1.0001:
        return 1.0
        
    def area_mach_residual(M):
        if M == 0:
            return 1e6
        term1 = 2.0 / (gamma + 1.0)
        term2 = 1.0 + (gamma - 1.0) / 2.0 * M**2
        return (1.0 / M) * (term1 * term2)**((gamma + 1.0) / (2.0 * (gamma - 1.0))) - AR
        
    if is_subsonic:
        return brentq(area_mach_residual, 1e-5, 1.0)
    else:
        return brentq(area_mach_residual, 1.0, 15.0)

# --- Load Coolant Data from DWSIM ---
print("Loading Methanol/Water NRTL data...")
try:
    df = pd.read_excel('MethanolWaterData8atm.xlsx')
    df.columns = df.columns.str.strip() 
    
    if df['Viscosity'].mean() > 0.05: 
        df['Viscosity'] = df['Viscosity'] * 0.001
        print(" -> Viscosity auto-converted from cP to Pa-s")
        
    if df['Cp'].mean() < 15.0:
        df['Cp'] = df['Cp'] * 1000.0
        print(" -> Cp auto-converted from kJ/kg-K to J/kg-K")

    get_rho = interp1d(df['Temperature'], df['Density'], kind='linear', fill_value="extrapolate")
    get_mu = interp1d(df['Temperature'], df['Viscosity'], kind='linear', fill_value="extrapolate")
    get_cp = interp1d(df['Temperature'], df['Cp'], kind='linear', fill_value="extrapolate")
    get_k = interp1d(df['Temperature'], df['Conductivity'], kind='linear', fill_value="extrapolate")
except Exception as e:
    print(f"Error loading Excel data: {e}")
    sys.exit(1)

# --- Load Gas Emissivity Database ---
print("Loading Gas Emissivity Database...")
try:
    emissivity_df = pd.read_csv('gas_emissivity_grid.csv')
    points = emissivity_df[['Temperature_K', 'Path_Length_cm']].values
    values = emissivity_df['Broadband_Emissivity'].values
    get_emissivity = LinearNDInterpolator(points, values)
    sigma_sb = 5.67e-8  # Stefan-Boltzmann Constant (W/m^2*K^4)
except Exception as e:
    print(f"Error loading Emissivity CSV: {e}")
    sys.exit(1)

# --- Initial Conditions ---
R_t = (3.7e-2) / 2          # Throat Radius in m
A_t = np.pi * R_t**2        # Throat area in m^2
P_c = 6 * 101325            # Chamber pressure in Pa
P_e = 1 * 101325            # Exit pressure in Pa
ContractionRatio = 9        # A_c/A_t
L_star = 1.5                # m
MR = 0.95326                # Mixture Ratio (O/F)

# --- Cooling Jacket Parameters ---
T_coolant_in = 293.15       # Coolant inlet temperature at nozzle exit (K)
P_coolant = 8 * 101325      # Coolant pressure (8 atm)
t_wall = 0.0015             # Wall thickness (m)
k_wall = 160                # Thermal conductivity for AlSi10Mg

# --- 3D Printed Channel Geometry ---
N_channels = 22             # Number of discrete printed cooling channels
w_ch = 0.0013               # Channel width: 1.3 mm 
h_ch = 0.0013               # Channel height: 1.3 mm

# 1. Define the custom thermodynamic card for the 70/30 blend
blend_card = """
fuel CH3OH(L) C 1 H 4 O 1 wt%=70.0
h,cal=-57103.767 t(k)=298
fuel water H 2.0 O 1.0 wt%=30.0
h,cal=-68317.732 t(k)=298 
"""

# 2. Inject the custom fuel into RocketCEA's backend database
add_new_fuel('Methox30', blend_card)

# 3. Initialize your engine using the new custom fuel name
engine = CEA_Obj(
    oxName='GOX', fuelName='Methox30', 
    pressure_units='MPa', cstar_units='m/s', 
    temperature_units='K', isp_units='sec', sonic_velocity_units='m/s'
)

Pc_MPa = P_c / 1e6
Pe_MPa = P_e / 1e6
PcOvPe = Pc_MPa / Pe_MPa

# Extract thermodynamic constraints from CEA
eps_exit = engine.get_eps_at_PcOvPe(Pc=Pc_MPa, MR=MR, PcOvPe=PcOvPe)
T_c, T_t, T_e_cea = engine.get_Temperatures(Pc=Pc_MPa, MR=MR, eps=eps_exit)
mw_c, gamma_c_cea = engine.get_Chamber_MolWt_gamma(Pc=Pc_MPa, MR=MR, eps=eps_exit)
Cstar_cea = engine.get_Cstar(Pc=Pc_MPa, MR=MR)

# Calculate Mass Flows
filmcoolpercent = 20 # %
m_dot_coolant = P_c*A_t/((Cstar_cea)*(1+MR))*(1+filmcoolpercent/100)  
m_dot_oxidizer = MR*m_dot_coolant/(1+filmcoolpercent/100) 
m_dot_film = m_dot_coolant*(filmcoolpercent/100) # Explicit film coolant mass flow

Vol_Chamber = L_star * A_t     
A_c = ContractionRatio * A_t   
A_e = eps_exit * A_t                
L_chamber = Vol_Chamber / A_c  
R_c = np.sqrt(A_c / np.pi)     
R_e = np.sqrt(A_e / np.pi)     

def prandtl_meyer(M, gamma):
    term1 = np.sqrt((gamma + 1) / (gamma - 1))
    term2 = np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (M**2 - 1)))
    term3 = np.arctan(np.sqrt(M**2 - 1))
    return term1 * term2 - term3

M_e_cea = engine.get_MachNumber(Pc=Pc_MPa, MR=MR, eps=eps_exit)
nu_max = (np.pi / 2) * (np.sqrt((gamma_c_cea + 1) / (gamma_c_cea - 1)) - 1)
nu_M = prandtl_meyer(M_e_cea, gamma_c_cea)
ConeAngleMax = np.degrees(nu_max - nu_M)
DivergingConeAngle = ConeAngleMax if ConeAngleMax < 10 else 10
ConvergingConeAngle = 45 

rConverging = 1.5 * R_t
rDiverging = 0.382 * R_t

xtanconverging = L_chamber + rConverging * cosd(acotd(tand(ConvergingConeAngle)))
ytanconverging = R_c - rConverging + rConverging * sind(acotd(tand(ConvergingConeAngle)))
ytanconvergingthroat = R_t + rConverging - rConverging * sind(acotd(tand(ConvergingConeAngle)))
xconvergingcone = (ytanconverging - ytanconvergingthroat) / tand(ConvergingConeAngle)
xtanconvergingthroat = xtanconverging + xconvergingcone + rConverging * cosd(acotd(tand(ConvergingConeAngle)))
xtanDiverging = xtanconvergingthroat + rDiverging * cosd(acotd(tand(DivergingConeAngle)))
ytanDiverging = R_t + rDiverging - rDiverging * sind(acotd(tand(DivergingConeAngle)))
xDivergingCone = xtanDiverging + (R_e - ytanDiverging) / tand(DivergingConeAngle)

N_points = 80 # Discrete axial stations for PDE solver
x = np.linspace(0, xDivergingCone, N_points)
y = np.zeros_like(x)

for i, xi in enumerate(x):
    if xi <= L_chamber:
        y[i] = R_c
    elif xi <= xtanconverging:
        y[i] = R_c - rConverging + np.sqrt(max(rConverging**2 - (xi - L_chamber)**2, 0))
    elif xi <= xtanconverging + xconvergingcone:
        y[i] = -tand(ConvergingConeAngle) * xi + ytanconverging + tand(ConvergingConeAngle) * xtanconverging
    elif xi <= xtanconvergingthroat:
        y[i] = R_t + rConverging - np.sqrt(max(rConverging**2 - (xi - xtanconvergingthroat)**2, 0))
    elif xi <= xtanDiverging:
        y[i] = R_t + rDiverging - np.sqrt(max(rDiverging**2 - (xi - xtanconvergingthroat)**2, 0))
    else:
        y[i] = tand(DivergingConeAngle) * xi + ytanDiverging - tand(DivergingConeAngle) * xtanDiverging

x_throat = xtanconvergingthroat
eps_array = (y**2) / (R_t**2)

print("Evaluating Core Flow (Pure Frozen Isentropic)...")
gas = ct.Solution('gri30.yaml')

mass_fraction_O2 = MR / (1.0 + MR)
mass_fraction_fuel_total = 1.0 / (1.0 + MR)
mass_fraction_CH3OH = 0.70 * mass_fraction_fuel_total
mass_fraction_H2O = 0.30 * mass_fraction_fuel_total
gas.Y = f"CH3OH:{mass_fraction_CH3OH}, H2O:{mass_fraction_H2O}, O2:{mass_fraction_O2}"
gas.TP = T_c, P_c
gas.equilibrate('TP') 
chamber_X = gas.X     

Mach = np.zeros(N_points)
T_gas = np.zeros(N_points)
P_gas = np.zeros(N_points)
U_gas = np.zeros(N_points)

# Pre-allocate core property arrays for the PDE
rho_e_array = np.zeros(N_points)
mu_e_array = np.zeros(N_points)
H_e_array = np.zeros(N_points)
core_X_array = []

for i in range(N_points):
    AR = max(eps_array[i], 1.0001)
    is_subsonic = x[i] <= x_throat
    
    Mach[i] = solve_mach_from_area(AR, gamma_c_cea, is_subsonic)
    T_gas[i] = T_c / (1 + (gamma_c_cea - 1)/2 * Mach[i]**2)
    P_gas[i] = P_c / (1 + (gamma_c_cea - 1)/2 * Mach[i]**2)**(gamma_c_cea / (gamma_c_cea - 1))
    
    gas.TPX = T_gas[i], P_gas[i], chamber_X
    U_gas[i] = Mach[i] * gas.sound_speed
    
    rho_e_array[i] = gas.density
    mu_e_array[i] = gas.viscosity
    H_e_array[i] = gas.enthalpy_mass
    core_X_array.append(chamber_X)

print("\n--- GLOBAL CHT SOLVER (PDE + RADIATION) ---")
max_global_iterations = 150  
cht_tolerance = 0.5  
T_bulk_array = np.linspace(T_coolant_in + 50, T_coolant_in, N_points)
T_wg_array = np.linspace(800, 400, N_points) 
q_flux_pde = np.zeros(N_points)

M_eta = 40                 
eta_max = 8.0              
eta = np.linspace(0, eta_max, M_eta)
d_eta = eta[1] - eta[0]

a_diag = np.zeros(M_eta); b_diag = np.zeros(M_eta); c_diag = np.zeros(M_eta); d_rhs = np.zeros(M_eta)

for iter_global in range(max_global_iterations):
    if iter_global % 10 == 0:
        print(f"Global CHT Iteration {iter_global+1}...")
    
    T_wg_old_global = np.copy(T_wg_array)
    
    # --- 1. FORWARD MARCH: 2D PDE GAS SOLVER ---
    xi = np.zeros(N_points)
    U_old = 1.0 - np.exp(-eta) 
    H_old = np.ones(M_eta) * H_e_array[0] 
    f_old = np.zeros(M_eta)
    for j in range(1, M_eta):
        f_old[j] = f_old[j-1] + 0.5 * (U_old[j] + U_old[j-1]) * d_eta

    for i in range(1, N_points):
        dx_step = x[i] - x[i-1]
        
        # Illingworth-Levy coordinates & Pressure Gradient
        integrand_prev = rho_e_array[i-1] * U_gas[i-1] * mu_e_array[i-1]
        integrand_curr = rho_e_array[i] * U_gas[i] * mu_e_array[i]
        d_xi = 0.5 * (integrand_curr + integrand_prev) * dx_step
        xi[i] = xi[i-1] + d_xi
        
        beta = (2 * xi[i] / U_gas[i]) * ((U_gas[i] - U_gas[i-1]) / d_xi)
        beta = np.clip(beta, -0.5, 2.0) 

        # ========================================================
        # LIQUID FILM VAPORIZATION & GASEOUS DECAY MODEL
        # ========================================================
        T_film_inject = T_bulk_array[0] # Extracted dynamically from injector manifold
        
        # Thermodynamics of 70/30 Methanol/Water at 5 atm 
        # ---> UPDATE THESE WITH EXACT DWSIM VALUES ONCE EXCEL IS FINISHED <---
        T_sat_film = 405.0         
        Cp_liquid = 3300.0         
        h_fg_film = 1.4e6          
        Cp_vapor = 1790          
        k_c_vapor = 0.0268561 
        rho_c_vapor = 3.89          
        
        if i == 1:
            h_gas_prev = 1000.0
            accumulated_heat = 0.0 
            x_virtual_origin = 0.0 
            film_is_liquid = True
        else:
            delta_T = T_gas[i-1] - T_wg_array[i-1]
            h_gas_prev = q_conv / delta_T if delta_T > 0 else 1000.0
            h_gas_prev = np.clip(h_gas_prev, 100.0, 50000.0)
            accumulated_heat += q_conv * dx_step * (np.pi * 2.0 * y[i])

        Q_boil_threshold = m_dot_film * (Cp_liquid * (T_sat_film - T_film_inject) + h_fg_film)

        if accumulated_heat < Q_boil_threshold and film_is_liquid:
            # Phase 1: Liquid Length
            eta_film = 1.0 
            T_aw = T_sat_film 
            x_virtual_origin = x[i] 
        else:
            # Phase 2: Gaseous Decay
            if film_is_liquid:
                if iter_global == max_global_iterations - 1:
                    print(f"   -> Film vaporized at x = {x[i]*100:.2f} cm")
                film_is_liquid = False
                
            x_decay = max(x[i] - x_virtual_origin, 0.0)
            V_gas = U_gas[i]
            V_film = 15.0  # Approx vapor velocity
            
            # Classic f(Vr) piecewise function
            if V_gas / V_film >= 1.0:
                f_Vr = 1.0 + 0.4 * np.arctan((V_gas / V_film) - 1.0)
            else:
                f_Vr = (V_film / V_gas)**1.5  
                
            # Local Circumference Correction
            local_circumference = 2.0 * np.pi * y[i]
            m_dot_film_prime_local = m_dot_film / local_circumference
            term_1 = (h_gas_prev * x_decay) / (m_dot_film_prime_local * Cp_vapor)
            
            # Radiation-Corrected K and Thermal Diffusivity Term
            K_rad = -0.018 
            alpha_c = k_c_vapor / (rho_c_vapor * Cp_vapor)
            s_slot = 0.001  
            term_2 = K_rad * ((s_slot * V_film) / alpha_c)**0.125 * f_Vr
            
            eta_film = np.exp(-(term_1 + term_2))
            eta_film = np.clip(eta_film, 0.0, 1.0)
            T_aw = T_gas[i] - eta_film * (T_gas[i] - T_sat_film)

        # Throat Boundary Layer Stripping
        if x[i] >= x_throat:
            eta_film = 0.0  
            T_aw = T_gas[i] 

        # ========================================================
        # PDE SOLVER CORE LOGIC
        # ========================================================
        C_array = np.zeros(M_eta); E_array = np.zeros(M_eta); rho_array = np.zeros(M_eta)
        
        # Override Edge Enthalpy with Adiabatic Wall Temp (Film cooling)
        gas.TPX = T_aw, P_gas[i], core_X_array[i]
        H_edge = gas.enthalpy_mass
        
        T_w = T_wg_array[i]
        
        gas.TPX = T_w, P_gas[i], core_X_array[i]
        H_wg = gas.enthalpy_mass
        C_wall = (gas.density * gas.viscosity) / (rho_e_array[i] * mu_e_array[i])
        Pr_w = (gas.cp_mass * gas.viscosity) / gas.thermal_conductivity
        
        for j in range(M_eta):
            T_j = T_w + (T_gas[i] - T_w) * U_old[j]
            gas.TPX = T_j, P_gas[i], core_X_array[i] 
            rho_array[j] = gas.density
            mu_j = gas.viscosity
            Pr_j = (gas.cp_mass * mu_j) / gas.thermal_conductivity
            C_array[j] = (rho_array[j] * mu_j) / (rho_e_array[i] * mu_e_array[i])
            E_array[j] = C_array[j] / Pr_j

        U_guess = np.copy(U_old)
        for iteration in range(5):
            f_curr = np.zeros(M_eta)
            for j in range(1, M_eta): f_curr[j] = f_curr[j-1] + 0.5 * (U_guess[j] + U_guess[j-1]) * d_eta
                
            for j in range(1, M_eta - 1):
                dC_deta = (C_array[j+1] - C_array[j-1]) / (2 * d_eta)
                K1 = C_array[j] / (d_eta**2)
                K2 = (1 / (2 * d_eta)) * (dC_deta + f_curr[j] + 2 * xi[i] * ((f_curr[j] - f_old[j]) / d_xi))
                K3 = beta * U_guess[j] + (2 * xi[i] * U_guess[j]) / d_xi
                a_diag[j] = K1 - K2; b_diag[j] = -2 * K1 - K3; c_diag[j] = K1 + K2
                d_rhs[j] = -((2 * xi[i] * U_guess[j]) / d_xi) * U_old[j] - beta * (rho_e_array[i] / rho_array[j])
                
            b_diag[0] = 1.0; d_rhs[0] = 0.0
            b_diag[-1] = 1.0; d_rhs[-1] = 1.0

            ab_U = np.zeros((3, M_eta))
            ab_U[0, 1:] = c_diag[:-1]; ab_U[1, :] = b_diag; ab_U[2, :-1] = a_diag[1:]
            U_new = solve_banded((1, 1), ab_U, d_rhs)
            
            if np.max(np.abs(U_new - U_guess)) < 1e-4: break
        U_guess = np.copy(U_new)

        for j in range(1, M_eta - 1):
            dE_deta = (E_array[j+1] - E_array[j-1]) / (2 * d_eta)
            K1_g = E_array[j] / (d_eta**2)
            K2_g = (1 / (2 * d_eta)) * (dE_deta + f_curr[j] + 2 * xi[i] * ((f_curr[j] - f_old[j]) / d_xi))
            K3_g = (2 * xi[i] * U_new[j]) / d_xi
            a_diag[j] = K1_g - K2_g; b_diag[j] = -2 * K1_g - K3_g; c_diag[j] = K1_g + K2_g
            d_rhs[j] = -K3_g * H_old[j]

        b_diag[0] = 1.0; d_rhs[0] = H_wg 
        b_diag[-1] = 1.0; d_rhs[-1] = H_edge
        
        ab_g = np.zeros((3, M_eta))
        ab_g[0, 1:] = c_diag[:-1]; ab_g[1, :] = b_diag; ab_g[2, :-1] = a_diag[1:]
        H_new = solve_banded((1, 1), ab_g, d_rhs)

        dH_deta_wall = (-3*H_new[0] + 4*H_new[1] - H_new[2]) / (2 * d_eta)
        q_conv = (rho_e_array[i] * U_gas[i] * mu_e_array[i]) / (np.sqrt(2 * xi[i]) * Pr_w) * C_wall * dH_deta_wall

        # ========================================================
        # RADIATION HEAT FLUX CALCULATION
        # ========================================================
        L_e_cm = 0.9 * (2.0 * y[i] * 100.0)
        T_gas_safe = np.clip(T_gas[i], 1000.0, 3500.0)
        L_e_safe = np.clip(L_e_cm, 0.1, 100.0)
        
        epsilon_gas = float(get_emissivity(T_gas_safe, L_e_safe))
        q_rad = sigma_sb * epsilon_gas * (T_gas[i]**4 - T_w**4)
        
        q_flux_pde[i] = q_conv + q_rad

        U_old = np.copy(U_new)
        H_old = np.copy(H_new)
        f_old = np.copy(f_curr)
    
    q_flux_pde[0] = q_flux_pde[1] 

    # --- 2. BACKWARD MARCH: COOLANT & WALL UPDATE ---
    T_bulk_curr = T_coolant_in
    for i in range(N_points - 1, -1, -1):
        D_x = 2 * y[i]
        dx = x[i] - x[i-1] if i > 0 else x[1] - x[0]
        
        rho_c = float(get_rho(T_bulk_curr))
        mu_c = float(get_mu(T_bulk_curr))
        cp_c = float(get_cp(T_bulk_curr))
        k_c = float(get_k(T_bulk_curr))
        
        D_h_cool = (2 * w_ch * h_ch) / (w_ch + h_ch)  
        A_flow_cool = N_channels * w_ch * h_ch        
        V_cool = m_dot_coolant / (rho_c * A_flow_cool)
        Re_cool = (rho_c * V_cool * D_h_cool) / mu_c
    
        if Re_cool <= 3000:
            print(f"Warning: Laminar flow detected at x={x[i]:.3f} m (Re={Re_cool:.1f})")
            sys.exit(1)

        Pr_cool = (cp_c * mu_c) / k_c
        
        f_friction = (0.79 * np.log(Re_cool) - 1.64)**-2
        Nu_cool = ((f_friction/8) * (Re_cool - 1000) * Pr_cool) / (1 + 12.7 * np.sqrt(f_friction/8) * (Pr_cool**(2/3) - 1))
        h_cool = (Nu_cool * k_c) / D_h_cool
    
        dT_bulk = (q_flux_pde[i] * np.pi * D_x * dx) / (m_dot_coolant * cp_c)
        T_bulk_curr += dT_bulk
        
        T_bulk_curr = np.clip(T_bulk_curr, 275.0, 440.0) 

        T_bulk_array[i] = T_bulk_curr
        T_wg_new = T_bulk_array[i] + q_flux_pde[i] * (1/h_cool + t_wall/k_wall)
       
        omega = 0.08  
        T_wg_array[i] = (1.0 - omega) * T_wg_array[i] + omega * T_wg_new

    # --- 3. CONVERGENCE CHECK ---
    max_temp_change = np.max(np.abs(T_wg_array - T_wg_old_global))
    if iter_global % 10 == 0 or max_temp_change < cht_tolerance:
        print(f"   Max Wall Temp Change: {max_temp_change:.2f} K")
    
    if max_temp_change < cht_tolerance:
        print(f"\n>> CHT Solver Converged in {iter_global + 1} iterations! <<")
        break
else:
    print("\n>> Warning: CHT Solver hit max iterations before convergence. <<")

print("\n--- NAVIER-STOKES PDE ANALYSIS COMPLETE ---")
print(f"Coolant Inlet Temp (Nozzle Exit)  : {T_coolant_in - 273.15:.2f} °C")
print(f"Coolant Outlet Temp (At Injector) : {T_bulk_array[0] - 273.15:.2f} °C")
print(f"Peak Wall Temperature (PDE Solved): {np.max(T_wg_array[30:np.size(T_wg_array)]) - 273.15:.2f} °C")
print(f"Peak Wall Heat Flux (PDE Solved)  : {np.max(q_flux_pde[30:np.size(q_flux_pde)])/1e6:.2f} MW/m^2")
print(f"Need {30*m_dot_coolant:.2f} kg fuel for 30 second burn")
print(f"Need {30*m_dot_oxidizer:.2f} kg oxygen for 30 second burn")

throatmaxribthickness=np.sqrt(8*k_wall*t_wall*(250+273.15-np.max(T_wg_array[30:np.size(T_wg_array)]))/np.max(q_flux_pde[30:np.size(q_flux_pde)]))
chambermaxribthickness=np.sqrt(8*k_wall*t_wall*(250+273.15-np.max(T_wg_array[10:40]))/np.max(q_flux_pde[10:40]))
throatribthickness=(2*np.pi*(R_t+t_wall)-N_channels*w_ch)/N_channels
chamberribthickness=(2*np.pi*(R_c+t_wall)-N_channels*w_ch)/N_channels
minwallthickness=P_coolant*R_c/80e6
minFuelannulusarea=3*N_channels*w_ch*h_ch
Isp=engine.get_Isp(Pc=Pc_MPa, MR=MR, eps=eps_exit)
Thrust=P_c*A_t/(Cstar_cea)*engine.get_Isp(Pc=Pc_MPa, MR=MR, eps=eps_exit)*9.81

print(f"Max Throat Rib Thickness for Aluminum Wall: {throatmaxribthickness*1000:.2f} mm")
print(f"Max Chamber Rib Thickness for Aluminum Wall: {chambermaxribthickness*1000:.2f} mm")
print(f"Throat Rib Thickness for Aluminum Wall: {throatribthickness*1000:.2f} mm")
print(f"Chamber Rib Thickness for Aluminum Wall: {chamberribthickness*1000:.2f} mm")
print(f"Minimum Wall Thickness for Aluminum Wall: {minwallthickness*1000:.2f} mm")
print(f"Minimum Fuel Annulus Area for Aluminum Wall: {minFuelannulusarea*1e6:.2f} mm^2")
print(f"Predicted Isp: {Isp:.2f} s")
print(f"Predicted Thrust: {Thrust:.2f} N")

fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# 1. Nozzle Geometry
axs[0].plot(x * 100, y * 100, color='black', linewidth=2, label='Wall Contour')
axs[0].plot(x * 100, -y * 100, color='black', linewidth=2)
axs[0].axvline(x_throat * 100, color='red', linestyle='--', alpha=0.5, label='Throat')
axs[0].set_ylabel('Radius (cm)')
axs[0].set_title('Nozzle Geometry')
axs[0].legend()
axs[0].grid(True)

# 2. Final Temperature Profiles
axs[1].plot(x * 100, T_gas, color='orange', label='Core Gas Temp (K)')
axs[1].plot(x * 100, T_wg_array, color='red', label='Wall Temp (Gas-Side, K)')
axs[1].plot(x * 100, T_bulk_array, color='blue', label='Coolant Bulk Temp (K)')
axs[1].axvline(x_throat * 100, color='gray', linestyle='--', alpha=0.5)
axs[1].set_ylabel('Temperature (K)')
axs[1].set_title('Thermal Profiles (Frozen Core Flow)')
axs[1].legend()
axs[1].grid(True)

# 3. Heat Flux Comparison
axs[2].plot(x * 100, q_flux_pde / 1e6, color='purple', linewidth=2, label='PDE Heat Flux (Exact)')
axs[2].axvline(x_throat * 100, color='gray', linestyle='--', alpha=0.5)
axs[2].set_ylabel('Heat Flux (MW/m²)')
axs[2].set_xlabel('Axial Position (cm)')
axs[2].set_title('Regenerative Cooling Heat Flux Load')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()
# %%