# 200-lbf Liquid Rocket Engine: Thermal Analysis & CAD Suite

**Overview:** A computational thermal modeling pipeline and 3D-printable hardware design (AlSi10Mg) for a pressure-fed liquid rocket engine utilizing Gaseous Oxygen (GOX) and a 70/30 Methanol-Water fuel blend.

![CAD Cross Section Placeholder](link-to-your-onshape-screenshot.png)

## 1. System Specifications
* **Thrust:** 200 lbf (Horizontal Static Fire Configuration)
* **Chamber Pressure ($P_c$):** 5–6 atm
* **Propellants:** GOX / 70% Methanol - 30% Water ($O/F = 0.95$)
* **Adiabatic Flame Temp:** ~2800 K
* **Chamber Material:** Direct Metal Laser Sintered (DMLS) AlSi10Mg

## 2. Computational Thermal Pipeline (Python)
To evaluate steady-state survivability of the AlSi10Mg chamber walls, a custom 2D PDE boundary-layer solver was developed in Python:
* **Automated Property Integration:** Coupled **NASA CEA** and **DWSIM** to dynamically extract freestream combustion products, non-ideal mixture thermodynamics, bubble points, and enthalpy of vaporization across varying temperature regimes.
* **Broad-Spectrum Radiative Heat Transfer:** Integrated **RADIS** with the HITEMP database (0–8000 $\text{cm}^{-1}$) to generate lookup tables via `scipy.interpolate` for gas emissivity across $\text{H}_2\text{O}$, $\text{CO}_2$, and $\text{CO}$ combustion species.
* **Regenerative & Film Cooling:** Modeled co-current regenerative cooling channel heat pickup alongside a 20% **Hatch-Papell correlation** film cooling boundary condition and 5% faceplate transpiration cooling.

![Thermal Plot Placeholder](link-to-your-temperature-plot.png)

## 3. Hardware Design & Trade Studies (Onshape CAD)
* **Injector Architecture:** Evaluated pintle and single-element coaxial designs before selecting a **7-element hexagonal shear coaxial array**. Pulling the outer 6 elements inward prevented impingement on the regenerative cooling channels while ensuring optimal atomization.
* **Faceplate Thermal Management:** Allocated 5% of fuel mass flow through six 0.46 mm weep holes for faceplate transpiration cooling.
* **Combustion Stability:** Integrated 17.4 mm deep quarter-wave acoustic resonator cavities along the outer injector perimeter to damp high-frequency acoustic modes.

## 4. Repository Structure
* `thermal_solver.py` — Primary 2D PDE boundary layer and regenerative cooling script.
* `radis_emissivity.py` — Spectral radiation script generating gas emissivity CSV lookup tables.
* `/cad_exports` — STEP files and engineering drawings of the chamber and injector assembly.
