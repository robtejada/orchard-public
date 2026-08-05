# BCtable – 4D Atmospheric Boundary Condition Interpolator

This module provides quadlinear (4D) interpolation of atmospheric boundary-condition data stored in a compressed `.npz` datacube. Data is defined over a 4D parameter space:

(T_eff, log g, Y, Z)

where:

- T_eff = effective temperature [K]
- log g = surface gravity (cgs)
- Y = helium mass fraction
- Z = metallicity in solar


## 1. Loading the Datacube

The class reads the compressed `spectra_grid_YZmerged.npz` file containing the atmospheric grid.

```python
from BCtable import BCtable

bc = BCtable("spectra_grid_YZmerged.npz")
```


## 2. Datacube Structure

The Class BCtable read from `.npz` file contains attributes:

### Grids

- `Teff_grid` : Effective temperature grid [K]
- `logg_grid` : log g grid
- `logY_grid` : Helium fraction grid
- `logZ_grid` : Metallicity (in solar) grid
- `lam` : Wavelength grid [micron]


### Data Arrays

- `entropy` : shape (N_T, N_g, N_Y, N_Z)
- `flux` : shape (N_T, N_g, N_Y, N_Z, N_lam)
- `T_profile` : shape (N_T, N_g, N_Y, N_Z, 99)
- `P_profile` : shape (N_T, N_g, N_Y, N_Z, 99)



## 3. Interpolating a Single Spectrum

```python
lam, F = bc.interpolate_spectrum_4D(
    Teff_in,
    logg_in,
    Y_in,
    Z_in
)
```

Inputs:

- Teff_in : Effective temperature [K]
- logg_in : log g
- Y_in : Helium fraction (> 0)
- Z_in : Metallicity (> 0)

Output:

- lam : wavelength grid [micron]
- F : interpolated spectrum [erg cm^-2 s^-1 Hz^-1]


## 4. Spectral Evolution Along a Track

```python
lam, all_flux = bc.spectrum_evolution(
    N_frames,
    age_path,
    Teff_path,
    logg_path,
    Z_path,
    Y_path,
    plot=False,
    figname="spectrum_evolution.pdf"
)
```

### Purpose

Computes the time evolution of the emergent spectrum along an evolutionary trajectory in parameter space.

---

### Required Inputs

- `N_frames`  
  Integer. Number of snapshots along the evolutionary track.

- `age_path`  
  1D array of length `N_frames`.  
  Age in **Gyr (gigayears)**.  
  Used for time-color mapping if `plot=True`.

- `Teff_path`  
  1D array of length `N_frames`.  
  Effective temperature in **Kelvin**.

- `logg_path`  
  1D array of length `N_frames`.  
  Surface gravity in **log10 g (cgs)**.

- `Y_path`  
  1D array of length `N_frames`.  
  Helium mass fraction (must be > 0).

- `Z_path`  
  1D array of length `N_frames`.  
  Metallicity (must be > 0).

All arrays must:

- Have identical length equal to `N_frames`
- Correspond element-wise to the same evolutionary snapshot
- Be within the bounds of the tabulated grid (values are clipped if slightly outside)

---

### Optional Arguments

- `plot` (default: `False`)  
  If `True`, generates:
  - 3D helium evolution plot
  - 3D metallicity evolution plot
  - Spectral evolution panel

- `figname`  
  Output filename if plotting is enabled.

---

### Returns

- `lam`  
  Wavelength grid in microns. Shape `(N_lam,)`.

- `all_flux`  
  Interpolated spectra. Shape `(N_frames, N_lam)`.

  `all_flux[i]` corresponds to the spectrum at:
  ```
  age_path[i],
  Teff_path[i],
  logg_path[i],
  Y_path[i],
  Z_path[i]
  ```
---

### Special Case

If `age_path` is a scalar (single snapshot), the function returns:

- `lam`
- `all_flux` with shape `(1, N_lam)`

and skips evolution plotting.

## 5. Example Usage

```python
import numpy as np
from BCtable import BCtable

bc = BCtable("BC_grid_YZmerged_FULL.npz")

N = 10
age = np.logspace(-3, 1, N)
Teff = np.linspace(1400, 100, N)
logg = np.linspace(3.0, 4.2, N)
Y = np.linspace(0.275, 0.15, N)
Z = np.linspace(3.16, 5.00, N)

lam, spectra = bc.spectrum_evolution(
    N,
    age,
    Teff,
    logg,
    Z,
    Y,
    plot=True
)
```

produces the wavelength range and spectra with shape `(N, N_lam)`.  An example figure is generated showing the spectral evolution over time.