"""Lightweight model loader for notebook workflows.
Example usage:

paths = [
    "parameters/parameter_user_super_jupiter_10mj.ini",
    "parameters/parameter_user_super_jupiter_10mj_nozcond.ini",
    "parameters/parameter_user_super_jupiter_10mj_norad_nozcond.ini",
]

models = load_models.ModelSet(paths, verbose=True)
# or: models = load_models.load_models(paths)

# Access like plot_class-style lists:
T0 = models.temp[0]
P0 = models.press[0]
age0 = models.data_age[0]

# Or per-model dicts:
m0 = models[0] # model data for 0th model 
rho0 = m0["rho"] # density data for 0th model
meta0 = models.metadata[0]

Developer: Roberto Tejada Arevalo (APPLE+ORCHARD)
"""

from __future__ import annotations

from contextlib import ExitStack
import configparser
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional
import warnings

import h5py
import numpy as np


def get_arith_mean(x: np.ndarray) -> np.ndarray:
    return (x[1:] + x[:-1]) / 2.0


SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0

_AGE_UNIT_TO_SECONDS = {
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "yr": SECONDS_PER_YEAR,
    "yrs": SECONDS_PER_YEAR,
    "year": SECONDS_PER_YEAR,
    "years": SECONDS_PER_YEAR,
    "myr": 1.0e6 * SECONDS_PER_YEAR,
    "mya": 1.0e6 * SECONDS_PER_YEAR,
    "megayear": 1.0e6 * SECONDS_PER_YEAR,
    "megayears": 1.0e6 * SECONDS_PER_YEAR,
    "gyr": 1.0e9 * SECONDS_PER_YEAR,
    "gya": 1.0e9 * SECONDS_PER_YEAR,
    "gigayear": 1.0e9 * SECONDS_PER_YEAR,
    "gigayears": 1.0e9 * SECONDS_PER_YEAR,
}


def _age_units_scale(units: str = "Gyr") -> float:
    key = str(units).strip().lower()
    if key not in _AGE_UNIT_TO_SECONDS:
        raise ValueError(
            f"Unsupported age units '{units}'. "
            "Supported examples: 's', 'year', 'Myr', 'Gyr'."
        )
    return _AGE_UNIT_TO_SECONDS[key]


def _age_to_seconds(age_value: float, units: str = "Gyr") -> float:
    return float(age_value) * _age_units_scale(units)


def _seconds_to_age(seconds: Any, units: str = "Gyr") -> np.ndarray:
    return np.asarray(seconds, dtype=float) / _age_units_scale(units)


def _pick_age_index(times_seconds: Any, target_seconds: float, mode: str = "nearest_left") -> int:
    """Return an index into a monotonically increasing time array."""
    arr = np.asarray(times_seconds, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot select an age from an empty time array.")

    mode_key = str(mode).strip().lower()
    if mode_key in ("left", "ceil", "ge", "first_ge"):
        idx = int(np.searchsorted(arr, target_seconds, side="left"))
        return int(np.clip(idx, 0, arr.size - 1))
    if mode_key in ("right", "floor", "le", "last_le"):
        idx = int(np.searchsorted(arr, target_seconds, side="right") - 1)
        return int(np.clip(idx, 0, arr.size - 1))

    idx_right = int(np.searchsorted(arr, target_seconds, side="left"))
    idx_right = int(np.clip(idx_right, 0, arr.size - 1))
    idx_left = int(np.clip(idx_right - 1, 0, arr.size - 1))

    dl = abs(arr[idx_left] - target_seconds)
    dr = abs(arr[idx_right] - target_seconds)

    if mode_key in ("nearest",):
        return idx_right if dr <= dl else idx_left
    if mode_key in ("nearest_left", "nearest-earlier-on-tie", "nearest_left_tie"):
        return idx_right if dr < dl else idx_left

    raise ValueError(
        f"Unsupported age selection mode '{mode}'. "
        "Use 'nearest_left' (default), 'nearest', 'floor', or 'ceil'."
    )


class AgeArray(np.ndarray):
    """NumPy array subclass with age-based indexing via ``.age(...)``.

    Works like a normal ndarray for arithmetic/indexing, but stores the model's
    age grid (in seconds) so users can request snapshots in physical units.
    """

    def __new__(
        cls,
        input_array: Any,
        *,
        age_seconds: Any,
        field_name: str,
        model_index: Optional[int] = None,
        time_axis: int = 0,
    ) -> "AgeArray":
        obj = np.asarray(input_array).view(cls)
        obj._age_seconds = np.asarray(age_seconds, dtype=float).reshape(-1)
        obj._field_name = str(field_name)
        obj._model_index = model_index
        obj._time_axis = int(time_axis)
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        self._age_seconds = getattr(obj, "_age_seconds", None)
        self._field_name = getattr(obj, "_field_name", None)
        self._model_index = getattr(obj, "_model_index", None)
        self._time_axis = getattr(obj, "_time_axis", 0)

    def ages(self, units: str = "Gyr") -> np.ndarray:
        """Return the available age grid in the requested units."""
        return _seconds_to_age(self._age_seconds, units=units)

    def age_index(self, age_value: float, units: str = "Gyr", mode: str = "nearest_left") -> int:
        """Return the selected snapshot index for a requested age."""
        target_s = _age_to_seconds(age_value, units=units)
        idx = _pick_age_index(self._age_seconds, target_s, mode=mode)

        if self.ndim == 0:
            return 0
        if self._time_axis >= self.ndim:
            raise ValueError(
                f"Field '{self._field_name}' does not have a valid time axis "
                f"(time_axis={self._time_axis}, ndim={self.ndim})."
            )

        n_time = int(self.shape[self._time_axis])
        if n_time <= 0:
            raise ValueError(f"Field '{self._field_name}' has an empty time axis.")
        return int(np.clip(idx, 0, n_time - 1))

    def age(
        self,
        age_value: float,
        *,
        units: str = "Gyr",
        mode: str = "nearest_left",
        return_age: bool = False,
        return_index: bool = False,
    ) -> Any:
        """Return the value/profile at the requested age.

        Parameters
        ----------
        age_value : float
            Requested age.
        units : str, default ``'Gyr'``
            Age units. Supported examples: ``s``, ``year``, ``Myr``, ``Gyr``.
        mode : str, default ``'nearest_left'``
            Selection rule when the exact age is absent.
        return_age : bool
            If True, also return the matched age (in the requested units).
        return_index : bool
            If True, also return the matched time index.
        """
        idx = self.age_index(age_value, units=units, mode=mode)

        if self.ndim == 0:
            value = np.asarray(self).item()
        elif self.ndim == 1:
            value = np.asarray(self)[idx]
        else:
            value = np.take(np.asarray(self), idx, axis=self._time_axis)

        matched_age = float(_seconds_to_age(self._age_seconds[idx], units=units))

        if return_age and return_index:
            return value, matched_age, idx
        if return_age:
            return value, matched_age
        if return_index:
            return value, idx
        return value

    def __repr__(self) -> str:
        base = np.ndarray.__repr__(self)
        field = getattr(self, "_field_name", None)
        if field is None:
            return base
        return f"AgeArray(field={field!r}, shape={self.shape}, dtype={self.dtype})\n{base}"


class ModelSnapshot:
    """Snapshot of one model at a selected age (dict-like + attribute access)."""

    def __init__(
        self,
        data: Dict[str, Any],
        *,
        model_index: int,
        snapshot_index: int,
        requested_age: float,
        requested_units: str,
        matched_age_seconds: float,
    ) -> None:
        self._data = data
        self.model_index = int(model_index)
        self.snapshot_index = int(snapshot_index)
        self.requested_age = float(requested_age)
        self.requested_units = str(requested_units)
        self.matched_age_seconds = float(matched_age_seconds)

    @property
    def age_seconds(self) -> float:
        return self.matched_age_seconds

    def age_in(self, units: str = "Gyr") -> float:
        return float(_seconds_to_age(self.matched_age_seconds, units=units))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            return self._data[name]
        raise AttributeError(name)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def __repr__(self) -> str:
        return (
            f"ModelSnapshot(model_index={self.model_index}, snapshot_index={self.snapshot_index}, "
            f"age={self.age_in('Gyr'):.6f} Gyr, keys={len(self._data)})"
        )


class ModelView:
    """User-facing handle for one model with age-aware access helpers."""

    def __init__(self, parent: "ModelSet", index: int) -> None:
        self._parent = parent
        self.index = int(index)

    @property
    def _data(self) -> Dict[str, Any]:
        return self._parent.models[self.index]

    @property
    def meta(self) -> Dict[str, Any]:
        return self._parent.metadata[self.index]

    @property
    def param_path(self) -> Optional[str]:
        return self._parent.param_paths[self.index] if self.index < len(self._parent.param_paths) else None

    @property
    def model_dir(self) -> Optional[str]:
        return self._parent.model_dirs[self.index] if self.index < len(self._parent.model_dirs) else None

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            return self._data[name]
        raise AttributeError(name)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()

    def ages(self, units: str = "Gyr") -> np.ndarray:
        age_arr = self._data.get("data_age")
        if isinstance(age_arr, AgeArray):
            return age_arr.ages(units=units)
        if age_arr is None:
            raise ValueError("Model has no 'data_age' array.")
        return _seconds_to_age(age_arr, units=units)

    def snapshot(
        self,
        age_value: float,
        *,
        units: str = "Gyr",
        mode: str = "nearest_left",
    ) -> ModelSnapshot:
        """Return a snapshot object with all fields evaluated at the requested age."""
        age_arr = self._data.get("data_age")
        if not isinstance(age_arr, AgeArray):
            raise ValueError("Model does not have an age-aware 'data_age' array.")

        _, matched_age_units, idx = age_arr.age(
            age_value,
            units=units,
            mode=mode,
            return_age=True,
            return_index=True,
        )
        matched_age_seconds = float(age_arr.age(age_value, units=units, mode=mode))

        snap: Dict[str, Any] = {}
        for key, val in self._data.items():
            if isinstance(val, AgeArray):
                snap[key] = val.age(age_value, units=units, mode=mode)
            else:
                snap[key] = val

        snap["meta"] = dict(self.meta)
        snap["param_path"] = self.param_path
        snap["model_dir"] = self.model_dir
        snap["requested_age"] = float(age_value)
        snap["requested_units"] = str(units)
        snap["matched_age"] = float(matched_age_units)
        snap["matched_age_seconds"] = matched_age_seconds
        snap["snapshot_index"] = int(idx)

        return ModelSnapshot(
            snap,
            model_index=self.index,
            snapshot_index=idx,
            requested_age=age_value,
            requested_units=units,
            matched_age_seconds=matched_age_seconds,
        )

    # alias with a more descriptive name for notebooks
    at_age = snapshot

    def age(
        self,
        age_value: float,
        *,
        units: str = "Gyr",
        mode: str = "nearest_left",
        field: str = "temp",
    ) -> Any:
        """Convenience shortcut: return one field at a given age.

        By default this returns the temperature profile (``field='temp'``), so
        ``models[i].age(3.3)`` gives the temperature profile at ~3.3 Gyr.
        Use ``models[i].snapshot(...)`` / ``models[i].at_age(...)`` to get all
        fields at once.
        """
        if field not in self._data:
            raise KeyError(f"Unknown field '{field}'.")
        value = self._data[field]
        if not isinstance(value, AgeArray):
            raise TypeError(f"Field '{field}' is not age-dependent.")
        return value.age(age_value, units=units, mode=mode)

    def __repr__(self) -> str:
        return f"ModelView(index={self.index}, model_dir={self.model_dir!r})"


class ModelSet:
    """Load one or more model outputs and expose arrays for analysis.

    Notes
    -----
    - ``paths`` should be parameter-file paths (same inputs used by
      ``plot_class.init_plot``).
    - The loader derives the model directory as ``models/<param-stem>`` with any
      embedded output suffix (``.hdf5``, ``.h5``, ``.npz``) removed.
    - Data are accessible both as per-model dicts (``models[i]``) and as
      list attributes (e.g. ``models.temp[i]``).
    """

    # Keys returned by _read_one_model (kept explicit for stable attributes)
    _MODEL_KEYS = [
        "mass_grid",
        "mass_norm",
        "mass_norm_i",
        "rad",
        "press",
        "rho",
        "temp",
        "S",
        "Y",
        "Z",
        "Ym",
        "Z1m",
        "Z2m",
        "f_rock",
        "conv_flux",
        "rad_flux",
        "semi_flux",
        "Lambda_cond_i",
        "Lambda_rad_i",
        "bracket_dsdr",
        "dsdy_prho",
        "dsdz_prho",
        "dsdy_pt",
        "dsdz_pt",
        "brunt",
        "R0_inv",
        "mantle_phases",
        "data_age",
        "data_kcore",
        "data_kcore_fe",
        "data_tint",
        "data_teff",
        "data_grav",
        "data_rad",
        "data_omega",
        "data_Y_atm",
        "data_Z_atm",
        "luminosity",
        "J2_TOF",
        "J4_TOF",
        "J6_TOF",
        "oblat_TOF",
        "Req_TOF",
        "Rpol_TOF",
        "Rvol_TOF",
        "data_I",
        "deltaE",
        "dE_S",
        "energy",
        "melt_fraction",
    ]
    _TIME_DEPENDENT_KEYS = {
        "rad",
        "press",
        "rho",
        "temp",
        "S",
        "Y",
        "Z",
        "Ym",
        "Z1m",
        "Z2m",
        "f_rock",
        "conv_flux",
        "rad_flux",
        "semi_flux",
        "Lambda_cond_i",
        "Lambda_rad_i",
        "bracket_dsdr",
        "dsdy_prho",
        "dsdz_prho",
        "dsdy_pt",
        "dsdz_pt",
        "brunt",
        "R0_inv",
        "mantle_phases",
        "data_age",
        "data_kcore",
        "data_kcore_fe",
        "data_tint",
        "data_teff",
        "data_grav",
        "data_rad",
        "data_omega",
        "data_Y_atm",
        "data_Z_atm",
        "luminosity",
        "J2_TOF",
        "J4_TOF",
        "J6_TOF",
        "oblat_TOF",
        "Req_TOF",
        "Rpol_TOF",
        "Rvol_TOF",
        "data_I",
        "deltaE",
        "dE_S",
        "energy",
        "melt_fraction",
    }

    def __init__(
        self,
        paths: Optional[Iterable[str]] = None,
        *,
        verbose: bool = False,
        read_verbose: Optional[bool] = None,
    ) -> None:
        if isinstance(read_verbose, str):
            rv = read_verbose.strip().lower()
            if rv in ("", "auto", "infer"):
                read_verbose = None
            elif rv in ("1", "true", "yes", "y"):
                read_verbose = True
            elif rv in ("0", "false", "no", "n"):
                read_verbose = False
            else:
                raise ValueError(
                    "read_verbose must be True/False or 'auto'/'infer'."
                )

        self.param_paths: List[str] = list(paths or [])
        self.model_dirs: List[str] = [self._model_dir_from_param(p) for p in self.param_paths]
        # Compatibility with plot_class.init_plot, which stores model directories in self.paths.
        self.paths: List[str] = list(self.model_dirs)
        self.verbose = verbose
        self.read_verbose = read_verbose
        self.read_verbose_modes: List[bool] = []

        # Metadata parsed from parameter files (kept as list-of-dicts and list attrs).
        self.metadata: List[Dict[str, Any]] = []
        self._init_metadata_lists()
        self._parse_all_parameter_files()

        # Data containers (list-of-arrays per field).
        self.models: List[Dict[str, Any]] = []
        for key in self._MODEL_KEYS:
            setattr(self, key, [])
        # Alias for plot_class naming
        self.oblateness_TOF: List[Any] = []

        if self.verbose and self.paths:
            print("Reading files...")

        for model_idx, model_dir in enumerate(self.paths):
            d = self._read_one_model(model_dir)
            self.read_verbose_modes.append(bool(d.get("read_verbose", False)))
            d = self._wrap_age_arrays(d, model_idx=model_idx)
            self.models.append(d)
            self._append_model_data(d)

        if self.verbose and self.paths:
            print("Finished reading files...")

        # Optional convenience arrays commonly used in analysis notebooks.
        self.m_unit: List[np.ndarray] = []
        self.m_mean: List[np.ndarray] = []
        self.kcore: List[int] = []
        self.kcore_fe: List[int] = []
        self._compute_basic_helpers()

    # --------------------------
    # Public convenience methods
    # --------------------------
    def __len__(self) -> int:
        return len(self.models)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.models)

    def __getitem__(self, idx: int) -> ModelView:
        return ModelView(self, idx)

    def get(self, idx: int) -> Dict[str, Any]:
        """Return a shallow copy of one model dict plus metadata."""
        out = dict(self.models[idx])
        out["meta"] = dict(self.metadata[idx]) if idx < len(self.metadata) else {}
        out["param_path"] = self.param_paths[idx] if idx < len(self.param_paths) else None
        out["model_dir"] = self.model_dirs[idx] if idx < len(self.model_dirs) else None
        return out

    def as_list(self) -> List[Dict[str, Any]]:
        """Return list of per-model dicts (same objects stored internally)."""
        return self.models

    # --------------------------
    # Metadata parsing
    # --------------------------
    def _init_metadata_lists(self) -> None:
        self.misc_curve: List[Optional[str]] = []
        self.deltaT: List[Optional[float]] = []
        # Z-misc stand-in channel (zmisc_hhe=True, or legacy misc_kind='hhe'):
        # effective offsets with the transport.py fallbacks applied
        # (zmisc_* -> misc_*), plus an activity flag.
        self.zmisc_deltaT: List[Optional[float]] = []
        self.zmisc_deltaP: List[float] = []
        self.zmisc_active: List[bool] = []
        self.input_eos: List[Optional[str]] = []
        self.fixed_y: List[Optional[bool]] = []
        self.misc_pad: List[Optional[bool]] = []
        self.mcore_e: List[Optional[float]] = []
        self.mcore: List[Optional[float]] = []
        self.mcore_fe_e: List[Optional[float]] = []
        self.mcore_fe: List[Optional[float]] = []
        self.m_factor: List[Optional[float]] = []
        self.planet: List[Optional[str]] = []
        self.R0: List[Optional[float]] = []
        self.final_age: List[Optional[float]] = []
        self.mantle_comp: List[Optional[str]] = []
        self.eos_mantle: List[Optional[str]] = []
        self.eos_core: List[Optional[str]] = []

    @staticmethod
    def _cfg_get(cfg: configparser.ConfigParser, section: str, key: str, cast: str = "str") -> Any:
        if section not in cfg or key not in cfg[section]:
            return None
        if cast == "str":
            return str(cfg[section][key])
        if cast == "float":
            return float(cfg[section][key])
        if cast == "bool":
            return cfg[section].getboolean(key)
        raise ValueError(f"Unsupported cast: {cast}")

    def _parse_all_parameter_files(self) -> None:
        for path in self.param_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Parameter file not found: {path}")

            cfg = configparser.ConfigParser()
            cfg.read(path)

            # Resolve fractional core mass specification, mirroring the behavior
            # in utils.common._resolve_core_mass_fractions so that downstream
            # kcore fallbacks see the correct planet partition.
            mass_core_cfg = self._cfg_get(cfg, "core", "mass_core", "float")
            mass_core_fe_cfg = self._cfg_get(cfg, "core", "mass_core_fe", "float")
            if "core" in cfg and cfg["core"].getboolean("use_mass_fractions", fallback=False):
                M_Me = (self._cfg_get(cfg, "initial", "M_Mearth", "float")
                        or self._cfg_get(cfg, "initial", "M_factor", "float"))
                mantle_frac = self._cfg_get(cfg, "core", "mantle_fraction", "float") or 0.0
                iron_frac = self._cfg_get(cfg, "core", "iron_core_fraction", "float") or 0.0
                if M_Me is not None:
                    mass_core_cfg = mantle_frac * M_Me
                    mass_core_fe_cfg = iron_frac * mass_core_cfg

            md = {
                "param_path": path,
                "model_dir": self._model_dir_from_param(path),
                "misc_curve": self._cfg_get(cfg, "diffusion", "misc_curve", "str"),
                "misc_deltaT": self._cfg_get(cfg, "diffusion", "misc_deltaT", "float"),
                "misc_deltaP": self._cfg_get(cfg, "diffusion", "misc_deltaP", "float"),
                "zmisc_deltaT": self._cfg_get(cfg, "diffusion", "zmisc_deltaT", "float"),
                "zmisc_deltaP": self._cfg_get(cfg, "diffusion", "zmisc_deltaP", "float"),
                "metal_rain": self._cfg_get(cfg, "diffusion", "metal_rain", "bool"),
                "misc_kind": self._cfg_get(cfg, "diffusion", "misc_kind", "str"),
                "zmisc_hhe": self._cfg_get(cfg, "diffusion", "zmisc_hhe", "bool"),
                "hhe_eos": self._cfg_get(cfg, "equation_of_state", "hhe_eos", "str"),
                "misc_fixed_Y": self._cfg_get(cfg, "diffusion", "misc_fixed_Y", "bool"),
                "misc_pad": self._cfg_get(cfg, "diffusion", "misc_pad", "bool"),
                "mass_core": mass_core_cfg,
                "mass_core_fe": mass_core_fe_cfg,
                "mantle_comp": self._cfg_get(cfg, "core", "mantle_comp", "str"),
                "eos_mantle": self._cfg_get(cfg, "core", "eos_mantle", "str"),
                "eos_core": self._cfg_get(cfg, "core", "eos_core", "str"),
                "M_factor": self._cfg_get(cfg, "initial", "M_Mearth", "float") or self._cfg_get(cfg, "initial", "M_factor", "float"),
                "planet": self._cfg_get(cfg, "boundary_condition", "planet", "str"),
                "R_rho": self._cfg_get(cfg, "transport", "R_rho", "float"),
                "final_age": self._cfg_get(cfg, "general", "final_age", "float"),
            }
            self.metadata.append(md)

            # Compatibility list attrs used by some downstream notebooks/scripts.
            self.misc_curve.append(md["misc_curve"])
            self.deltaT.append(md["misc_deltaT"])
            # Effective Z-channel offsets, mirroring transport.py defaults
            self.zmisc_deltaT.append(md["zmisc_deltaT"] if md["zmisc_deltaT"] is not None
                                     else md["misc_deltaT"])
            self.zmisc_deltaP.append(md["zmisc_deltaP"] if md["zmisc_deltaP"] is not None
                                     else (md["misc_deltaP"] or 0.0))
            # Z stand-in is active when metal rain is on AND it uses the H-He
            # diagram — selected by zmisc_hhe (default True) or legacy
            # misc_kind == 'hhe'. zmisc_hhe defaults to True when unset.
            _zmisc_hhe = md["zmisc_hhe"] if md["zmisc_hhe"] is not None else True
            self.zmisc_active.append(
                bool(md["metal_rain"]) and (_zmisc_hhe or md["misc_kind"] == "hhe")
            )
            self.input_eos.append(md["hhe_eos"])
            self.fixed_y.append(md["misc_fixed_Y"])
            self.misc_pad.append(md["misc_pad"])
            self.mcore_e.append(md["mass_core"])
            self.mcore.append(md["mass_core"] * 0.00314558 if md["mass_core"] is not None else None)
            self.mcore_fe_e.append(md["mass_core_fe"])
            self.mcore_fe.append(md["mass_core_fe"] * 0.00314558 if md["mass_core_fe"] is not None else None)
            self.m_factor.append(md["M_factor"])
            self.planet.append(md["planet"])
            self.R0.append(md["R_rho"])
            self.final_age.append(md["final_age"])
            self.mantle_comp.append(md["mantle_comp"])
            self.eos_mantle.append(md["eos_mantle"])
            self.eos_core.append(md["eos_core"])

    def _append_model_data(self, d: Dict[str, Any]) -> None:
        for key in self._MODEL_KEYS:
            getattr(self, key).append(d.get(key))
        self.oblateness_TOF.append(d.get("oblat_TOF"))

    def _wrap_age_arrays(self, d: Dict[str, Any], *, model_idx: int) -> Dict[str, Any]:
        """Wrap time-dependent arrays as AgeArray while preserving ndarray behavior."""
        out = dict(d)
        age_seconds = out.get("data_age")
        if age_seconds is None:
            return out

        age_seconds_np = np.asarray(age_seconds, dtype=float).reshape(-1)
        if age_seconds_np.size == 0:
            return out

        for key in self._TIME_DEPENDENT_KEYS:
            val = out.get(key)
            if val is None:
                continue
            if isinstance(val, AgeArray):
                continue
            try:
                arr = np.asarray(val)
            except Exception:
                continue
            if arr.ndim == 0:
                continue
            if arr.size == 0 or arr.shape[0] == 0:
                continue
            # AgeArray assumes axis 0 is time (true for loader outputs).
            out[key] = AgeArray(
                arr,
                age_seconds=age_seconds_np,
                field_name=key,
                model_index=model_idx,
                time_axis=0,
            )
        return out

    def _compute_basic_helpers(self) -> None:
        for i in range(len(self.models)):
            mass_grid = self.mass_grid[i]
            if mass_grid is None or len(mass_grid) == 0:
                self.m_unit.append(np.array([]))
                self.m_mean.append(np.array([]))
                self.kcore.append(-1)
                self.kcore_fe.append(-1)
                continue

            mass_norm = self.mass_norm[i] if i < len(self.mass_norm) else None
            mass_norm_i = self.mass_norm_i[i] if i < len(self.mass_norm_i) else None

            if mass_norm is None or len(np.asarray(mass_norm)) == 0:
                mass_norm = np.asarray(mass_grid / mass_grid[0], dtype=float)
            else:
                mass_norm = np.asarray(mass_norm, dtype=float)

            if mass_norm_i is None or len(np.asarray(mass_norm_i)) == 0:
                mass_norm_i = np.asarray(get_arith_mean(mass_norm), dtype=float)
            else:
                mass_norm_i = np.asarray(mass_norm_i, dtype=float)

            # Keep legacy helper names used by plot_class.
            self.m_unit.append(mass_norm)
            self.m_mean.append(mass_norm_i)

            mcore_e = self.mcore_e[i] if i < len(self.mcore_e) else None
            mcore_fe_e = self.mcore_fe_e[i] if i < len(self.mcore_fe_e) else None

            # Prefer kcore / kcore_fe saved by the evolution writer (HDF5
            # metadata_json) over config-derived mass comparison.
            saved_kcore = self.models[i].get("kcore_saved") if i < len(self.models) else None
            saved_kcore_fe = self.models[i].get("kcore_fe_saved") if i < len(self.models) else None

            if saved_kcore is not None:
                self.kcore.append(int(saved_kcore))
            elif mcore_e is None or mcore_e <= 0:
                self.kcore.append(-1)
            else:
                mcore_cgs = mcore_e * 5.9722e27  # const.mearth, kept local to stay lightweight
                idx_core = np.where(self.mass_grid[i] < mcore_cgs)[0]
                self.kcore.append(int(idx_core[0]) if idx_core.size else -1)

            if saved_kcore_fe is not None:
                self.kcore_fe.append(int(saved_kcore_fe))
            elif mcore_fe_e is None or mcore_fe_e <= 0:
                self.kcore_fe.append(-1)
            else:
                mcore_fe_cgs = mcore_fe_e * 5.9722e27  # const.mearth
                idx_fe = np.where(self.mass_grid[i] < mcore_fe_cgs)[0]
                self.kcore_fe.append(int(idx_fe[0]) if idx_fe.size else -1)

    # --------------------------
    # I/O helpers: txt + HDF5
    # --------------------------
    def _find_h5_files(self, model_dir: str) -> List[str]:
        """Return a prioritized list of HDF5 files in ``model_dir``."""
        model_dir = str(model_dir)
        preferred = [
            "outputs.h5",
            "outputs.hdf5",
            "profiles.h5",
            "profiles.hdf5",
            "evolution.h5",
            "evolution.hdf5",
            "data.h5",
            "data.hdf5",
            "run.h5",
            "run.hdf5",
        ]

        files: List[str] = []
        for name in preferred:
            fp = os.path.join(model_dir, name)
            if os.path.isfile(fp):
                files.append(fp)

        extras = list(Path(model_dir).glob("*.h5")) + list(Path(model_dir).glob("*.hdf5"))
        files.extend(str(p) for p in sorted(extras))

        out: List[str] = []
        seen = set()
        for fp in files:
            if fp not in seen:
                out.append(fp)
                seen.add(fp)
        return out

    @staticmethod
    def _model_dir_from_param(param_path: str) -> str:
        """Map a parameter file path to a model directory under ``models/``."""
        name = Path(param_path).name

        if name.endswith(".ini"):
            name = name[:-4]

        for suf in (".hdf5", ".h5", ".npz"):
            if name.endswith(suf):
                name = name[: -len(suf)]

        return str(Path("models") / name)

    @staticmethod
    def _txt_read_array(fp: str) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"genfromtxt: Empty input file")
            return np.array(np.genfromtxt(fp))

    @staticmethod
    def _h5_get_exact(f: h5py.File, keys: Iterable[str]) -> Optional[np.ndarray]:
        """Try exact HDF5 paths/keys. Returns ``np.array(dataset)`` or ``None``."""
        for k in keys:
            kk = k.lstrip("/")
            if k in f:
                return np.array(f[k])
            if kk in f:
                return np.array(f[kk])
        return None

    @staticmethod
    def _h5_get_by_basename(f: h5py.File, basenames: Iterable[str]) -> Optional[np.ndarray]:
        """Search the HDF5 tree for a dataset with one of the requested basenames."""
        found: Optional[np.ndarray] = None
        basename_set = set(basenames)

        def visitor(name: str, obj: Any) -> None:
            nonlocal found
            if found is not None:
                return
            if isinstance(obj, h5py.Dataset):
                base = name.split("/")[-1]
                if base in basename_set:
                    found = np.array(obj)

        f.visititems(visitor)
        return found

    @staticmethod
    def _ensure_2d_time_by_zone(arr: Any, nz_guess: Optional[int] = None) -> Any:
        """Ensure profile-like arrays are ``(nt, nz)`` when shape can be inferred."""
        if arr is None:
            return None
        a = np.asarray(arr)
        if a.ndim == 1:
            return a
        if a.ndim != 2:
            return a

        if nz_guess is not None:
            if a.shape[1] == nz_guess:
                return a
            if a.shape[0] == nz_guess and a.shape[1] != nz_guess:
                return a.T
        return a

    def _read_one_model(self, model_dir: str) -> Dict[str, Any]:
        """Read one model directory, auto-detecting HDF5 vs legacy txt outputs."""
        model_dir = str(model_dir)
        out: Dict[str, Any] = {}

        h5_files = self._find_h5_files(model_dir)
        h5_handles: List[h5py.File] = []
        stack: Optional[ExitStack] = None

        if h5_files:
            stack = ExitStack()
            try:
                for fp in h5_files:
                    try:
                        h5_handles.append(stack.enter_context(h5py.File(fp, "r", swmr=True, libver="latest")))
                    except OSError:
                        # Skip unreadable HDF5 files.
                        pass
                if not h5_handles:
                    stack.close()
                    stack = None
            except Exception:
                stack.close()
                raise

        def _is_nonempty(val: Any) -> bool:
            if val is None:
                return False
            try:
                a = np.asarray(val)
            except Exception:
                return True
            return a.size > 0

        def h5_read(keys: Iterable[str] = (), basenames: Iterable[str] = (), default: Any = None) -> Any:
            if not h5_handles:
                return default
            for f in h5_handles:
                val = self._h5_get_exact(f, keys) if keys else None
                if (val is None or not _is_nonempty(val)) and basenames:
                    val = self._h5_get_by_basename(f, basenames)
                if val is None or not _is_nonempty(val):
                    continue
                return val
            return default

        def txt_path(name: str) -> str:
            return os.path.join(model_dir, name)

        def txt_exists(name: str) -> bool:
            return os.path.isfile(txt_path(name))

        # Read scalars the writer stores in the HDF5 metadata_json blob so
        # downstream consumers (kcore fallback in plot_class) can trust
        # saved partition indices over config-derived ones.
        saved_kcore = None
        saved_kcore_fe = None
        for f in h5_handles:
            if "metadata_json" in f:
                try:
                    import json as _json
                    meta_blob = _json.loads(f["metadata_json"][()])
                    if "kcore" in meta_blob:
                        saved_kcore = int(meta_blob["kcore"])
                    if "kcore_fe" in meta_blob:
                        saved_kcore_fe = int(meta_blob["kcore_fe"])
                except Exception:
                    pass
                if saved_kcore is not None:
                    break
        out["kcore_saved"] = saved_kcore
        out["kcore_fe_saved"] = saved_kcore_fe

        try:
            verbose_sentinel = h5_read(
                keys=(
                    "Lambda_cond_i",
                    "Lambda_rad_i",
                    "bracket_dsdr",
                    "transport/Lambda_cond_i",
                    "transport/Lambda_rad_i",
                    "transport/bracket_dsdr",
                    "profiles/Lambda_cond_i",
                    "profiles/Lambda_rad_i",
                    "profiles/bracket_dsdr",
                ),
                basenames=("Lambda_cond_i", "Lambda_rad_i", "bracket_dsdr"),
                default=None,
            )
            verbose_outputs_present = (
                verbose_sentinel is not None
                or txt_exists("Lambda_cond_i.txt")
                or txt_exists("Lambda_rad_i.txt")
                or txt_exists("bracket_dsdr.txt")
            )

            if self.read_verbose is None:
                read_verbose_mode = bool(verbose_outputs_present)
            elif bool(self.read_verbose) and not verbose_outputs_present:
                read_verbose_mode = False
                warnings.warn(
                    f"{model_dir}: verbose outputs not found; falling back to read_verbose=False.",
                    RuntimeWarning,
                )
            else:
                read_verbose_mode = bool(self.read_verbose)

            out["read_verbose"] = read_verbose_mode

            # ----------- read age early (helps orient time x zone arrays) -----------
            age_for_shape = h5_read(
                keys=("age", "evolution/age"),
                basenames=("age",),
                default=None,
            )
            nt_guess = len(np.asarray(age_for_shape)) if age_for_shape is not None else None

            # ----------- mass grid (needed for shape heuristics) -----------
            mgrid = h5_read(
                keys=(
                    "mgrid",
                    "grid/mgrid",
                    "mass_grid",
                    "grid/mass_grid",
                    "m_mean",
                    "grid/m_mean",
                    "m",
                    "m_b",
                ),
                basenames=("mgrid", "mass_grid", "m_mean", "m", "m_b"),
                default=None,
            )

            if mgrid is None:
                if txt_exists("mgrid.txt"):
                    mgrid = self._txt_read_array(txt_path("mgrid.txt"))
                else:
                    raise FileNotFoundError(
                        f"{model_dir}/mgrid.txt not found and no mgrid-like dataset was found in HDF5. "
                        f"HDF5 files detected: {h5_files}"
                    )

            mgrid = np.asarray(mgrid, dtype=float)

            if mgrid.ndim == 2:
                if nt_guess is not None and mgrid.shape[0] == nt_guess:
                    mgrid_1d = mgrid[-1, :]
                elif nt_guess is not None and mgrid.shape[1] == nt_guess:
                    mgrid_1d = mgrid[:, -1]
                else:
                    mgrid_1d = mgrid[-1, :]
                mgrid = mgrid_1d

            if mgrid.size == 0:
                raise ValueError(f"{model_dir}: mass grid is empty (mgrid/m_b dataset has size 0).")

            out["mass_grid"] = mgrid
            out["mass_norm"] = np.asarray(mgrid / mgrid[0], dtype=float)
            out["mass_norm_i"] = np.asarray(get_arith_mean(out["mass_norm"]), dtype=float)
            nz = len(out["mass_grid"])

            # ----------- profiles (2D: time x zone) -----------
            def read_profile(
                h5_keys: Iterable[str],
                h5_basenames: Iterable[str],
                txt_name: str,
                *,
                required: bool = True,
                nz_guess: Optional[int] = nz,
            ) -> Any:
                arr = h5_read(keys=h5_keys, basenames=h5_basenames, default=None)
                if arr is None and txt_exists(txt_name):
                    arr = self._txt_read_array(txt_path(txt_name))
                if arr is None and required:
                    raise FileNotFoundError(f"Missing {txt_name} and no matching HDF5 dataset in {model_dir}")
                return self._ensure_2d_time_by_zone(arr, nz_guess=nz_guess)

            out["rad"] = read_profile(
                ("rad_profiles", "profiles/rad", "rad", "r", "r_b", "profiles/r", "profiles/r_b"),
                ("rad_profiles", "rad", "r", "r_b"),
                "rad_profiles.txt",
                required=True,
                nz_guess=nz,
            )
            out["press"] = read_profile(
                ("press_profiles", "profiles/press", "press", "p"),
                ("press_profiles", "press", "p"),
                "press_profiles.txt",
                required=True,
                nz_guess=nz,
            )
            out["rho"] = read_profile(
                ("rho_profiles", "profiles/rho", "rho"),
                ("rho_profiles", "rho"),
                "rho_profiles.txt",
                required=True,
                nz_guess=nz,
            )
            out["temp"] = read_profile(
                ("temp_profiles", "profiles/temp", "temp"),
                ("temp_profiles", "temp"),
                "temp_profiles.txt",
                required=True,
                nz_guess=nz,
            )
            out["S"] = read_profile(
                ("Sprofiles", "profiles/S", "S"),
                ("Sprofiles", "S", "entropy"),
                "Sprofiles.txt",
                required=True,
                nz_guess=nz,
            )
            out["Y"] = read_profile(
                ("Yprofiles", "profiles/Y", "Y"),
                ("Yprofiles", "Y", "helium"),
                "Yprofiles.txt",
                required=True,
                nz_guess=nz,
            )
            out["Z"] = read_profile(
                ("Zprofiles", "profiles/Z", "Z"),
                ("Zprofiles", "Z", "metals"),
                "Zprofiles.txt",
                required=True,
                nz_guess=nz,
            )

            s_ref = np.asarray(out["S"])
            n_time_ref = int(s_ref.shape[0]) if s_ref.ndim == 2 else None
            n_cell_ref = int(s_ref.shape[-1]) if s_ref.ndim >= 1 else max(nz - 1, 1)
            n_bound_ref = int(np.asarray(out["mass_grid"]).shape[0])
            n_iface_ref = max(n_cell_ref - 1, 1)

            def _nan_time_profile(ncol: int) -> np.ndarray:
                if n_time_ref is None:
                    return np.full((ncol,), np.nan, dtype=float)
                return np.full((n_time_ref, ncol), np.nan, dtype=float)

            out["Ym"] = read_profile(
                ("Ym", "miscibility/Ym", "Ym"),
                ("Ym",),
                "Ym.txt",
                required=False,
                nz_guess=nz,
            )

            out["Z1m"] = read_profile(
                ("Z1m", "miscibility/Z1m", "Z1m"),
                ("Z1m",),
                "Z1m.txt",
                required=False,
                nz_guess=max(nz - 1, 1),
            )
            out["Z2m"] = read_profile(
                ("Z2m", "miscibility/Z2m", "Z2m"),
                ("Z2m",),
                "Z2m.txt",
                required=False,
                nz_guess=max(nz - 1, 1),
            )

            out["f_rock"] = read_profile(
                ("f_rock_profiles", "profiles/f_rock", "f_rock"),
                ("f_rock_profiles", "f_rock"),
                "f_rock_profiles.txt",
                required=False,
                nz_guess=nz,
            )

            if read_verbose_mode:
                out["conv_flux"] = read_profile(
                    ("conv_flux", "flux/conv", "profiles/conv_flux"),
                    ("conv_flux",),
                    "conv_flux.txt",
                    required=True,
                    nz_guess=n_bound_ref,
                )
            else:
                out["conv_flux"] = _nan_time_profile(n_bound_ref)
            if read_verbose_mode:
                out["rad_flux"] = read_profile(
                    ("rad_flux", "flux/rad", "profiles/rad_flux"),
                    ("rad_flux",),
                    "rad_flux.txt",
                    required=True,
                    nz_guess=n_bound_ref,
                )
            else:
                out["rad_flux"] = _nan_time_profile(n_bound_ref)
            if read_verbose_mode:
                out["semi_flux"] = read_profile(
                    ("semi_flux", "flux/semi", "profiles/semi_flux"),
                    ("semi_flux",),
                    "semi_flux.txt",
                    required=False,
                    nz_guess=n_bound_ref,
                )
            else:
                out["semi_flux"] = _nan_time_profile(n_bound_ref)
            if read_verbose_mode:
                out["Lambda_cond_i"] = read_profile(
                    ("Lambda_cond_i", "transport/Lambda_cond_i", "profiles/Lambda_cond_i"),
                    ("Lambda_cond_i",),
                    "Lambda_cond_i.txt",
                    required=False,
                    nz_guess=n_iface_ref,
                )
            else:
                out["Lambda_cond_i"] = _nan_time_profile(n_iface_ref)
            if read_verbose_mode:
                out["Lambda_rad_i"] = read_profile(
                    ("Lambda_rad_i", "transport/Lambda_rad_i", "profiles/Lambda_rad_i"),
                    ("Lambda_rad_i",),
                    "Lambda_rad_i.txt",
                    required=False,
                    nz_guess=n_iface_ref,
                )
            else:
                out["Lambda_rad_i"] = _nan_time_profile(n_iface_ref)
            if read_verbose_mode:
                out["bracket_dsdr"] = read_profile(
                    ("bracket_dsdr", "transport/bracket_dsdr", "profiles/bracket_dsdr"),
                    ("bracket_dsdr",),
                    "bracket_dsdr.txt",
                    required=False,
                    nz_guess=n_bound_ref,
                )
            else:
                out["bracket_dsdr"] = _nan_time_profile(n_bound_ref)

            if read_verbose_mode:
                out["dsdy_prho"] = read_profile(
                    ("dsdy_prho", "derivs/dsdy_prho"),
                    ("dsdy_prho",),
                    "dsdy_prho.txt",
                    required=True,
                    nz_guess=n_iface_ref,
                )
            else:
                out["dsdy_prho"] = _nan_time_profile(n_iface_ref)
            if read_verbose_mode:
                out["dsdz_prho"] = read_profile(
                    ("dsdz_prho", "derivs/dsdz_prho"),
                    ("dsdz_prho",),
                    "dsdz_prho.txt",
                    required=True,
                    nz_guess=n_iface_ref,
                )
            else:
                out["dsdz_prho"] = _nan_time_profile(n_iface_ref)
            if read_verbose_mode:
                out["dsdy_pt"] = read_profile(
                    ("dsdy_pt", "derivs/dsdy_pt"),
                    ("dsdy_pt",),
                    "dsdy_pt.txt",
                    required=True,
                    nz_guess=n_iface_ref,
                )
            else:
                out["dsdy_pt"] = _nan_time_profile(n_iface_ref)
            if read_verbose_mode:
                out["dsdz_pt"] = read_profile(
                    ("dsdz_pt", "derivs/dsdz_pt"),
                    ("dsdz_pt",),
                    "dsdz_pt.txt",
                    required=True,
                    nz_guess=n_iface_ref,
                )
            else:
                out["dsdz_pt"] = _nan_time_profile(n_iface_ref)

            out["brunt"] = read_profile(
                ("brunt", "profiles/brunt"),
                ("brunt",),
                "brunt.txt",
                required=True,
                nz_guess=max(nz - 1, 1),
            )

            out["R0_inv"] = read_profile(
                ("R0_inv", "transport/R0_inv", "profiles/R0_inv"),
                ("R0_inv",),
                "R0_inv.txt",
                required=True,
                nz_guess=n_bound_ref,
            )

            if read_verbose_mode:
                out["mantle_phases"] = read_profile(
                    ("mantle_phases", "profiles/mantle_phases"),
                    ("mantle_phases",),
                    "mantle_phases.txt",
                    required=False,
                    nz_guess=nz,
                )
            else:
                out["mantle_phases"] = None

            # ----------- timeseries (1D) -----------
            # Try combined evolution_data table first (HDF5 only).
            evo = h5_read(
                keys=("evolution_data", "evolution/data", "evolution"),
                basenames=("evolution_data", "evolution"),
                default=None,
            )
            # If no combined table, try per-field HDF5 datasets before falling
            # back to (potentially stale) text files.
            if evo is None:
                age = h5_read(keys=("age", "evolution/age"), basenames=("age",), default=None)
                Tint = h5_read(
                    keys=("Tint", "T_int", "tint"),
                    basenames=("Tint", "T_int", "tint"),
                    default=None,
                )
                Teff = h5_read(
                    keys=("Teff", "teff", "T_eff"),
                    basenames=("Teff", "teff", "T_eff"),
                    default=None,
                )
                grav = h5_read(keys=("grav", "g"), basenames=("grav", "g"), default=None)
                rad = h5_read(keys=("totrad", "radius", "R"), basenames=("totrad", "radius", "R"), default=None)
                lum = h5_read(
                    keys=("int_lum", "luminosity", "L_int"),
                    basenames=("int_lum", "luminosity", "L_int"),
                    default=None,
                )

                if age is not None and Tint is not None and Teff is not None and grav is not None and rad is not None:
                    n = len(np.asarray(age))
                    omega = h5_read(keys=("omega",), basenames=("omega",), default=np.full(n, np.nan))
                    evo = np.column_stack(
                        [
                            age,
                            Tint,
                            Teff,
                            grav,
                            rad,
                            np.full(n, np.nan),
                            omega,
                            lum if lum is not None else np.full(n, np.nan),
                        ]
                    )

            # Fall back to legacy text file only when HDF5 provided nothing.
            if evo is None and txt_exists("evolution_data.txt"):
                evo = np.genfromtxt(txt_path("evolution_data.txt"))

            if evo is None:
                raise FileNotFoundError(
                    f"Missing evolution_data(.txt or HDF5) in {model_dir}. "
                    "Looked for evolution_data OR per-field datasets (age, Tint, Teff, grav, totrad). "
                    f"HDF5 files detected: {h5_files}"
                )

            evo = np.asarray(evo)
            if evo.ndim == 1:
                evo = evo[None, :]

            out["data_age"] = np.asarray(evo[:, 0], dtype=float)
            out["data_tint"] = np.asarray(evo[:, 1], dtype=float)
            out["data_teff"] = np.asarray(evo[:, 2], dtype=float)
            out["data_grav"] = np.asarray(evo[:, 3], dtype=float)
            out["data_rad"] = np.asarray(evo[:, 4], dtype=float)
            out["data_omega"] = np.asarray(evo[:, 6], dtype=float) if evo.shape[1] > 6 else np.full(len(evo), np.nan)
            out["luminosity"] = np.asarray(evo[:, -1], dtype=float)

            # Atmosphere Y/Z histories are optional for backward compatibility.
            y_atm_raw = h5_read(
                keys=("Y_atm", "evolution/Y_atm"),
                basenames=("Y_atm",),
                default=None,
            )
            z_atm_raw = h5_read(
                keys=("Z_atm", "evolution/Z_atm"),
                basenames=("Z_atm",),
                default=None,
            )
            if (y_atm_raw is None or z_atm_raw is None) and txt_exists("evolution_yz.txt"):
                yz_tbl = np.asarray(self._txt_read_array(txt_path("evolution_yz.txt")))
                if yz_tbl.ndim == 1:
                    yz_tbl = yz_tbl[None, :]
                if y_atm_raw is None and yz_tbl.shape[1] > 0:
                    y_atm_raw = yz_tbl[:, 0]
                if z_atm_raw is None and yz_tbl.shape[1] > 1:
                    z_atm_raw = yz_tbl[:, 1]

            # ----------- J's (optional) -----------
            js = h5_read(keys=("Js_tof", "js_tof"), basenames=("Js_tof", "js_tof"), default=None)
            if js is None and txt_exists("Js_tof.txt"):
                js = np.genfromtxt(txt_path("Js_tof.txt"))

            nt = len(out["data_age"])

            def _to_timeseries(arr: Any) -> np.ndarray:
                if arr is None:
                    return np.full(nt, np.nan, dtype=float)
                a = np.asarray(arr, dtype=float).reshape(-1)
                if nt <= 0:
                    return a
                if a.size == nt:
                    return a
                out_ts = np.full(nt, np.nan, dtype=float)
                ncopy = min(nt, a.size)
                if ncopy > 0:
                    out_ts[:ncopy] = a[:ncopy]
                return out_ts

            out["data_Y_atm"] = _to_timeseries(y_atm_raw)
            out["data_Z_atm"] = _to_timeseries(z_atm_raw)

            if js is not None:
                js = np.asarray(js)
                if js.ndim == 1:
                    js = js[None, :]
                out["J2_TOF"] = np.asarray(js[:, 0], dtype=float)
                out["J4_TOF"] = np.asarray(js[:, 1], dtype=float)
                out["J6_TOF"] = np.asarray(js[:, 2], dtype=float) if js.shape[1] > 2 else np.full(len(js), np.nan)
                out["oblat_TOF"] = np.asarray(js[:, 4], dtype=float) if js.shape[1] > 4 else np.full(len(js), np.nan)
                out["Req_TOF"] = np.asarray(js[:, -3], dtype=float) if js.shape[1] >= 3 else np.full(len(js), np.nan)
                out["Rpol_TOF"] = np.asarray(js[:, -2], dtype=float) if js.shape[1] >= 2 else np.full(len(js), np.nan)
                out["Rvol_TOF"] = np.asarray(js[:, -1], dtype=float)
                out["data_I"] = np.asarray(js[:, -4], dtype=float) if js.shape[1] >= 4 else np.full(len(js), np.nan)
            else:
                # HDF5 runs store TOF outputs as individual scalar series.
                j2 = h5_read(keys=("j2", "evolution/j2"), basenames=("j2",), default=None)
                j4 = h5_read(keys=("j4", "evolution/j4"), basenames=("j4",), default=None)
                j6 = h5_read(keys=("j6", "evolution/j6"), basenames=("j6",), default=None)
                obl = h5_read(
                    keys=("obl", "oblat_TOF", "oblateness", "evolution/obl"),
                    basenames=("obl", "oblat_TOF", "oblateness"),
                    default=None,
                )
                req = h5_read(
                    keys=("eq_rad", "Req_TOF", "Req", "evolution/eq_rad"),
                    basenames=("eq_rad", "Req_TOF", "Req"),
                    default=None,
                )
                rpol = h5_read(
                    keys=("pol_rad", "Rpol_TOF", "Rpol", "evolution/pol_rad"),
                    basenames=("pol_rad", "Rpol_TOF", "Rpol"),
                    default=None,
                )
                rvol = h5_read(
                    keys=("Rvol_TOF", "rvol"),
                    basenames=("Rvol_TOF", "rvol"),
                    default=None,
                )
                moi = h5_read(
                    keys=("MoI", "I", "evolution/MoI"),
                    basenames=("MoI", "I"),
                    default=None,
                )

                if any(v is not None for v in (j2, j4, j6, obl, req, rpol, rvol, moi)):
                    req_ts = _to_timeseries(req)
                    rpol_ts = _to_timeseries(rpol)

                    out["J2_TOF"] = _to_timeseries(j2)
                    out["J4_TOF"] = _to_timeseries(j4)
                    out["J6_TOF"] = _to_timeseries(j6)
                    out["oblat_TOF"] = _to_timeseries(obl)
                    out["Req_TOF"] = req_ts
                    out["Rpol_TOF"] = rpol_ts
                    if rvol is None and req is not None and rpol is not None:
                        out["Rvol_TOF"] = np.power(req_ts, 2.0 / 3.0) * np.power(rpol_ts, 1.0 / 3.0)
                    else:
                        out["Rvol_TOF"] = _to_timeseries(rvol)
                    out["data_I"] = _to_timeseries(moi)
                else:
                    out["J2_TOF"] = None
                    out["J4_TOF"] = None
                    out["J6_TOF"] = None
                    out["oblat_TOF"] = None
                    out["Req_TOF"] = None
                    out["Rpol_TOF"] = None
                    out["Rvol_TOF"] = None
                    out["data_I"] = None

            # ----------- energies (optional) -----------
            ce = h5_read(keys=("conserv_energies", "energies/conserv"), basenames=("conserv_energies",), default=None)
            if ce is None and os.path.isfile(os.path.join(model_dir, "conserv_energies")):
                ce = np.genfromtxt(os.path.join(model_dir, "conserv_energies"))
            if ce is not None:
                ce = np.asarray(ce)
                if ce.ndim == 1:
                    ce = ce[None, :]
                out["deltaE"] = np.asarray(ce[:, 1], dtype=float) if ce.shape[1] > 1 else None
                out["dE_S"] = np.asarray(ce[:, 2], dtype=float) if ce.shape[1] > 2 else None
            else:
                out["deltaE"] = None
                out["dE_S"] = None

            en = h5_read(keys=("energies", "energies/total"), basenames=("energies",), default=None)
            if en is None and os.path.isfile(os.path.join(model_dir, "energies")):
                en = np.genfromtxt(os.path.join(model_dir, "energies"))
            out["energy"] = np.asarray(en) if en is not None else None

            # ----------- melt fraction (optional) -----------
            mf = h5_read(keys=("melt_fraction", "profiles/melt_fraction"), basenames=("melt_fraction",), default=None)
            if mf is None and txt_exists("melt_fraction.txt"):
                mf = self._txt_read_array(txt_path("melt_fraction.txt"))
            if mf is None:
                mf = np.full_like(out["S"], -1.0, dtype=float)
            out["melt_fraction"] = self._ensure_2d_time_by_zone(mf, nz_guess=nz)

            # ----------- mass-loss data (optional) -----------
            ml = h5_read(keys=("Mdot", "mass_loss/Mdot"), basenames=("Mdot",), default=None)
            if ml is not None:
                # HDF5 stores individual scalars per snapshot; assemble timeseries
                out["Mdot"] = np.atleast_1d(ml).flatten()
                m_env = h5_read(keys=("M_env",), basenames=("M_env",), default=None)
                out["M_env"] = np.atleast_1d(m_env).flatten() if m_env is not None else None
                m_pl = h5_read(keys=("M_planet_current",), basenames=("M_planet_current",), default=None)
                out["M_planet_current"] = np.atleast_1d(m_pl).flatten() if m_pl is not None else None
                l_adv = h5_read(keys=("L_adv",), basenames=("L_adv",), default=None)
                out["L_adv"] = np.atleast_1d(l_adv).flatten() if l_adv is not None else None
            elif txt_exists("mass_loss_data.txt"):
                ml_arr = self._txt_read_array(txt_path("mass_loss_data.txt"))
                if ml_arr is not None and ml_arr.ndim == 2 and ml_arr.shape[1] >= 5:
                    out["Mdot"] = ml_arr[:, 1]
                    out["M_env"] = ml_arr[:, 2]
                    out["M_planet_current"] = ml_arr[:, 3]
                    out["L_adv"] = ml_arr[:, 4]

            # ----------- per-step core boundary indices -----------
            # Time-resolved kcore(t) / kcore_fe(t) written by the evolution HDF5
            # saver.  Lets a run track a migrating boundary directly
            # (kcore can change over time, in the surface->centre index
            # convention) instead of re-deriving it from the Z profile.
            kc_series = h5_read(keys=("kcore",), basenames=("kcore",), default=None)
            out["data_kcore"] = (
                np.atleast_1d(kc_series).flatten().astype(float)
                if kc_series is not None else None
            )
            kcfe_series = h5_read(keys=("kcore_fe",), basenames=("kcore_fe",), default=None)
            out["data_kcore_fe"] = (
                np.atleast_1d(kcfe_series).flatten().astype(float)
                if kcfe_series is not None else None
            )

            return out
        finally:
            if stack is not None:
                stack.close()


def load(
    paths: Iterable[str],
    *,
    verbose: bool = True,
    read_verbose: Optional[bool] = None,
) -> ModelSet:
    """Convenience function returning a :class:`ModelSet`."""
    return ModelSet(paths=paths, verbose=verbose, read_verbose=read_verbose)


# Alias to mirror plot_class's "init_*" naming style.
init_models = ModelSet
