import numpy as np
import json
import os

try:
    import h5py
except ImportError:
    h5py = None


class H5EvolutionWriter:
    def __init__(
        self,
        path,
        *,
        compression="lzf",
        float_dtype="f4",
        flush_every=1,
        mode="w",
        swmr=True,
    ):
        if h5py is None:
            raise ImportError("h5py is required for save_format='hdf5'.")

        self.path = str(path)
        self.compression = compression
        self.float_dtype = float_dtype
        self.flush_every = max(1, int(flush_every))
        self._write_count = 0

        self._swmr_requested = bool(swmr)
        self._layout_finalized = False
        self._profile_names = None

        # Ensure parent exists
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

        # Open file
        self.f = h5py.File(self.path, mode, libver="latest")

        # Base datasets only (no shape assumptions)
        self._require_base_datasets()

        # If appending to an existing file, datasets already exist -> safe to enable SWMR now
        if self.f.mode in ("r+", "a") and self._has_layout():
            self._finalize_layout(enable_swmr=self._swmr_requested)

    def close(self):
        try:
            self.f.flush()
        finally:
            self.f.close()

    def _shuffle(self):
        # lzf is already fast; shuffle often helps gzip
        return (self.compression != "lzf")

    def _require_base_datasets(self):
        # 1D time-series scalars (length grows)
        for name, dtype in [
            ("age", "f8"), ("dt", "f8"), ("save_time", "f8"),
            ("T_int", "f4"), ("T_eff", "f4"), ("Y_atm", "f4"), ("Z_atm", "f4"), ("grav", "f4"),
            ("totrad", "f4"), ("mantle_rad", "f4"), ("core_rad", "f4"),
            ("kcore", "f8"), ("kcore_fe", "f8"),
            ("t10", "f4"), ("omega", "f4"), ("int_lum", "f8"),
            ("j2", "f8"), ("j4", "f8"), ("j6", "f8"), ("j8", "f8"),
            ("obl", "f8"), ("eq_rad", "f8"), ("pol_rad", "f8"), ("MoI", "f8"),
        ]:
            self.f.require_dataset(
                name, shape=(0,), maxshape=(None,), chunks=True, dtype=dtype,
                compression=self.compression, shuffle=self._shuffle()
            )

        # Energies / conservation tables
        self.f.require_dataset(
            "energies", shape=(0, 5), maxshape=(None, 5), chunks=(1, 5),
            dtype="f8", compression=self.compression, shuffle=self._shuffle()
        )
        self.f.require_dataset(
            "conserv_energies", shape=(0, 3), maxshape=(None, 3), chunks=(1, 3),
            dtype="f8", compression=self.compression, shuffle=self._shuffle()
        )

        if "metadata_json" not in self.f:
            self.f.create_dataset("metadata_json", shape=(), dtype=h5py.string_dtype("utf-8"))

    def _has_layout(self):
        # layout is considered present if at least one profile dataset exists
        return any(k in self.f for k in ["m_b", "r", "r_b", "p", "temp", "S"])

    def _finalize_layout(self, enable_swmr: bool):
        self.f.flush()
        self._layout_finalized = True
        if enable_swmr and self.f.mode in ("r+", "a", "w", "w-", "x"):
            self.f.swmr_mode = True

    def write_constants(self, metadata: dict):
        # Ignore m_b and dm here by design.
        self.f["metadata_json"][...] = json.dumps(metadata)
        self.f.flush()

    def _ensure_2d_series(self, name: str, ncol: int, dtype: str):
        """
        Ensure a (time, ncol) dataset exists.
        Must be called before SWMR is enabled if it needs to create a new dataset.
        """
        if name in self.f:
            d = self.f[name]
            if d.ndim != 2 or d.shape[1] != ncol:
                raise ValueError(
                    f"HDF5 dataset '{name}' shape mismatch: file has {d.shape}, "
                    f"but code is trying to write (time, {ncol})."
                )
            return d

        if self.f.swmr_mode:
            raise RuntimeError(
                f"Cannot create new dataset '{name}' after SWMR is enabled. "
                f"This indicates the first append didn't establish layout correctly."
            )

        return self.f.create_dataset(
            name,
            shape=(0, ncol),
            maxshape=(None, ncol),
            chunks=(1, ncol),
            dtype=dtype,
            compression=self.compression,
            shuffle=self._shuffle(),
        )

    def n_saved(self):
        return int(self.f["age"].shape[0])

    def load_last(self):
        n = self.n_saved()
        if n < 1:
            raise RuntimeError("HDF5 file has no saved snapshots to restart from.")
        i = n - 1

        # <<< CHANGED >>> m_b is now a profile; load it from the last snapshot.
        m_b_last = self.f["m_b"][i].astype(float) if "m_b" in self.f else None

        def _optional_scalar(name: str, default: float = np.nan) -> float:
            if name in self.f and len(self.f[name]) > i:
                return float(self.f[name][i])
            return float(default)

        out = {
            "dt_": float(self.f["dt"][i]),
            "t": float(self.f["age"][i]),
            "save_time": float(self.f["save_time"][i]),
            "T_int": float(self.f["T_int"][i]),
            "T_eff": float(self.f["T_eff"][i]),
            "Y_atm": _optional_scalar("Y_atm"),
            "Z_atm": _optional_scalar("Z_atm"),
            "omega": float(self.f["omega"][i]),
            "int_lum": float(self.f["int_lum"][i]),
            "totrad": float(self.f["totrad"][i]),
            "mantle_rad": float(self.f["mantle_rad"][i]),
            "core_rad": float(self.f["core_rad"][i]),
            "t10": float(self.f["t10"][i]),
            "grav": float(self.f["grav"][i]),
            "j2": float(self.f["j2"][i]) if "j2" in self.f else None,
            "j4": float(self.f["j4"][i]) if "j4" in self.f else None,
            "j6": float(self.f["j6"][i]) if "j6" in self.f else None,
            "j8": float(self.f["j8"][i]) if "j8" in self.f else None,
            "obl": float(self.f["obl"][i]) if "obl" in self.f else None,
            "eq_rad": float(self.f["eq_rad"][i]) if "eq_rad" in self.f else None,
            "pol_rad": float(self.f["pol_rad"][i]) if "pol_rad" in self.f else None,
            "MoI": float(self.f["MoI"][i]) if "MoI" in self.f else None,

            # grid/profile state
            "m_b": m_b_last,

            "temp_old": self.f["temp"][i].astype(float),
            "r_b_old": self.f["r_b"][i].astype(float),
            "p_old": self.f["p"][i].astype(float),
            "rho_old": self.f["rho"][i].astype(float),
            "S_old": self.f["S"][i].astype(float),
            "Y_old": self.f["Y"][i].astype(float),
            "Z_old": self.f["Z"][i].astype(float),
            "f_rock_old": self.f["f_rock"][i].astype(float),
        }

        n_cell = int(out["temp_old"].shape[0])
        n_bound = int(out["r_b_old"].shape[0])
        n_iface = max(n_cell - 1, 0)

        def _optional_profile(name: str, nout: int) -> np.ndarray:
            if name in self.f:
                return self.f[name][i].astype(float)
            return np.zeros((nout,), dtype=float)

        out["D_b_old"] = _optional_profile("Dcoeff", n_bound)
        out["conv_flux"] = _optional_profile("conv_flux", n_bound)
        out["rad_flux"] = _optional_profile("rad_flux", n_bound)
        out["semi_flux"] = _optional_profile("semi_flux", n_bound)
        out["dsdy_prho_i"] = _optional_profile("dsdy_prho", n_iface)
        out["dsdz_prho_i"] = _optional_profile("dsdz_prho", n_iface)
        out["dsdy_pt_i"] = _optional_profile("dsdy_pt", n_iface)
        out["dsdz_pt_i"] = _optional_profile("dsdz_pt", n_iface)

        # Optional convenience: if you used to rely on dm as constant, reconstruct it
        # (only if m_b looks sane and dm isn't explicitly stored).
        if out["m_b"] is not None and out["m_b"].ndim == 1 and out["m_b"].size >= 2:
            out["dm"] = np.diff(out["m_b"]).astype(float)

        return out

    def append(self, snap: dict):
        i = self.n_saved()

        required_scalar_keys = [
            "age","dt","save_time","T_int","T_eff","grav","totrad","mantle_rad","core_rad",
            "Y_atm","Z_atm","t10","omega","int_lum","j2","j4","j6","j8","obl","eq_rad","pol_rad","MoI",
            "energies_row","conserv_row"
        ]
        required_profile_keys = [
            # <<< CHANGED >>> m_b is a normal per-timestep profile now
            "m_b",
            "r","r_b","p","rho","temp","S","Y","Z","f_rock",
            "Ym","Z1m","Z2m",
            "brunt",
            "melt_fraction", "R0_inv",
        ]
        optional_profile_keys = [
            "conv_flux",
            "rad_flux",
            "semi_flux",
            "radio_flux",
            "Dcoeff",
            "D2",
            "cp",
            "dsdy_prho",
            "dsdy_pt",
            "dsdz_prho",
            "dsdz_pt",
            "Lambda_cond_i",
            "Lambda_rad_i",
            "bracket_dsdr",
        ]

        missing = [k for k in (required_scalar_keys + required_profile_keys) if k not in snap]
        if missing:
            raise KeyError(f"HDF5 snapshot missing keys: {missing}")

        if self._profile_names is None:
            if self._layout_finalized:
                profile_names = [k for k in (required_profile_keys + optional_profile_keys) if k in self.f]
            else:
                profile_names = required_profile_keys + [k for k in optional_profile_keys if k in snap]
            self._profile_names = list(profile_names)
        else:
            profile_names = list(self._profile_names)

        for k in optional_profile_keys:
            in_snap = (k in snap)
            in_layout = (k in profile_names)
            if in_snap and (not in_layout):
                raise KeyError(
                    f"Optional HDF5 profile '{k}' presence mismatch: "
                    f"in_snap={in_snap}, in_layout={in_layout}. "
                    "Keep snapshot keys consistent across the run."
                )

        # Create profile datasets on first append using actual shapes
        if not self._layout_finalized:
            for name in profile_names:
                arr = np.asarray(snap[name])
                if arr.ndim != 1:
                    raise ValueError(f"Expected 1D array for '{name}', got shape {arr.shape}")
                self._ensure_2d_series(name, arr.size, self.float_dtype)

            # Create optional scalar datasets before finalizing layout
            optional_scalars = ["Mdot", "M_env", "M_planet_current", "L_adv", "total_mass_lost"]
            for k in optional_scalars:
                if k in snap and k not in self.f:
                    self.f.create_dataset(
                        k, shape=(0,), maxshape=(None,),
                        dtype=self.float_dtype,
                        compression=self.compression,
                        shuffle=self._shuffle(),
                    )

            # Now safe to enable SWMR
            self._finalize_layout(enable_swmr=self._swmr_requested)

        # Scalars
        scalar_series = [
            "age","dt","save_time","T_int","T_eff","grav","totrad","mantle_rad","core_rad",
            "kcore","kcore_fe",
            "Y_atm","Z_atm","t10","omega","int_lum","j2","j4","j6","j8","obl","eq_rad","pol_rad","MoI"
        ]
        for k in scalar_series:
            d = self.f[k]
            n_prev = int(d.shape[0])
            d.resize((i + 1,))
            if n_prev < i:
                d[n_prev:i] = np.nan
            d[i] = snap[k]

        # Optional scalar series (mass loss, etc.)
        optional_scalars = ["Mdot", "M_env", "M_planet_current", "L_adv", "total_mass_lost"]
        for k in optional_scalars:
            if k in snap and k in self.f:
                d = self.f[k]
                n_prev = int(d.shape[0])
                d.resize((i + 1,))
                if n_prev < i:
                    d[n_prev:i] = np.nan
                d[i] = snap[k]

        # Energies
        self.f["energies"].resize((i + 1, 5))
        self.f["conserv_energies"].resize((i + 1, 3))
        self.f["energies"][i] = np.asarray(snap["energies_row"], dtype="f8")
        self.f["conserv_energies"][i] = np.asarray(snap["conserv_row"], dtype="f8")

        # Profiles
        for name in profile_names:
            d = self.f[name]
            if name in snap:
                arr = np.asarray(snap[name], dtype=self.float_dtype)
            else:
                arr = np.full((d.shape[1],), np.nan, dtype=self.float_dtype)
            d.resize((i + 1, d.shape[1]))
            if arr.shape != (d.shape[1],):
                raise ValueError(f"Shape mismatch for '{name}': got {arr.shape}, expected {(d.shape[1],)}")
            d[i] = arr

        self._write_count += 1
        if (self._write_count % self.flush_every) == 0:
            self.f.flush()
