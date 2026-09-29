from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, overload

import numpy as np
import pandas as pd
import xarray as xr

import pycontrails
from pycontrails.core.flight import Flight
from pycontrails.core.met import MetDataset
from pycontrails.core.met_var import (
    AirTemperature,
    EastwardWind,
    Geopotential,
    NorthwardWind,
    RelativeHumidity,
    SpecificHumidity,
)
from pycontrails.core.models import Model, ModelParams
from pycontrails.core.vector import GeoVectorDataset
from pycontrails.datalib import ecmwf


# Van Manen and Grewe 2019
DEFAULT_COEFFS = {
    "h2o": np.array([4.05e-16, 1.48e-16]),
    "o3": np.array([-5.20e-11, 2.30e-13, 4.85e-16, -2.04e-18]),
    "ch4": np.array([-9.83e-13, 1.99e-18, -6.32e-16, 6.12e-21]),
}


@dataclass
class ACCFParams(ModelParams):
    """ACCF model parameters."""

    # Versions & Efficacies
    version: str = "MATTHES_2023"
    efficacy: str = "DAHLMANN_2025"

    VALID_VERSIONS = ("VANMANEN_GREWE_2019", "YIN_DIETMUELLER_2023", "MATTHES_2023")
    VALID_EFFICACIES = ("LEE_2021", "DAHLMANN_2025", "NONE")

    scaling_factor = {
        "VANMANEN_GREWE_2019": {"CH4": 1.0, "O3": 1.0, "H2O": 1.0, "CiC": 1.0, "CO2": 1.0},
        "YIN_DIETMUELLER_2023": {"CH4": 2.03, "O3": 1.97, "H2O": 1.92, "CiC": 1.0, "CO2": 1.0},
        "MATTHES_2023": {"CH4": 35.0, "O3": 11.0, "H2O": 3.0, "CiC": 3.0, "CO2": 1.0},
    }

    efficacy_factors = {
        "LEE_2021": {"CH4": 1.18, "O3": 1.37, "H2O": 1.0, "CiC": 0.42},
        "DAHLMANN_2025": {"CH4": 1.04, "O3": 1.05, "H2O": 1.0, "CiC": 0.21},
        "NONE": {"CH4": 1.0, "O3": 1.0, "H2O": 1.0, "CiC": 1.0},
    }

    include_pmo: bool = True

    # Validity of aCCFs
    # horizontal
    lat_min: float = 30.0
    lat_max: float = 80.0
    lon_min: float = -75.0
    lon_max: float = 0.0
    # vertical
    air_pressure_min: float = 200.0 # hPa
    air_pressure_max: float = 350.0 # hPa


class ACCF(Model):
    """Compute Algorithmic Climate Change Functions (ACCF)."""

    name = "accf"
    long_name = "algorithmic climate change functions"
    met_variables = (
        AirTemperature,
        SpecificHumidity,
        ecmwf.PotentialVorticity,
        Geopotential,
        (RelativeHumidity, ecmwf.RelativeHumidity),
        NorthwardWind,
        EastwardWind,
    )
    sur_variables = (ecmwf.SurfaceSolarDownwardRadiation, ecmwf.TopNetThermalRadiation)
    default_params = ACCFParams

    def __init__(
        self,
        met: MetDataset,
        surface: MetDataset | None = None,
        params: dict[str, Any] | None = None,
        coefficients: dict[str, np.ndarray] | None = None,
        **params_kwargs: Any,
    ) -> None:
        # Normalize ECMWF variables
        variables = self.ecmwf_met_variables()
        met = met.standardize_variables(variables)

        # Convert RH percentage to proportion if needed
        if met["relative_humidity"].attrs.get("units") == "%":
            met.data["relative_humidity"] /= 100.0
            met.data["relative_humidity"].attrs["units"] = "1"

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="pycontrails.core.models")
            super().__init__(met, params=params, **params_kwargs)

        # Validate params
        if self.params["version"] not in ACCFParams.VALID_VERSIONS:
            raise ValueError(
                f"Invalid version: {self.params['version']}. "
                f"Must be one of {ACCFParams.VALID_VERSIONS}."
            )
        if self.params["efficacy"] not in ACCFParams.VALID_EFFICACIES:
            raise ValueError(
                f"Invalid efficacy: {self.params['efficacy']}. "
                f"Must be one of {ACCFParams.VALID_EFFICACIES}."
            )

        if surface:
            surface = surface.copy()
            surface = surface.standardize_variables(self.sur_variables)
            self.surface = surface

        self.coeffs = coefficients if coefficients is not None else DEFAULT_COEFFS

    def accf_h2o(self, potential_vorticity: np.ndarray | xr.DataArray) -> np.ndarray | xr.DataArray:
        """
        Determines the H2O aCCF.
        :param potential_vorticity: [K m2 kg-1 s-1]
        :return: aCCF-H2O in [K / kg(fuel)]
        """
        beta0, beta1 = self.coeffs["h2o"]

        accf = beta0 + beta1 * np.absolute(potential_vorticity * 1e6)  # pv * 1e6 = [PVU]
        accf = accf / ACCFParams.scaling_factor[self.params["version"]]["H2O"]
        accf = accf * ACCFParams.efficacy_factors[self.params["efficacy"]]["H2O"]

        if isinstance(accf, xr.DataArray):
            return accf.assign_attrs({
                "units": "K kg(fuel)**-1",
                "long_name": "algorithmic climate change function of water vapour",
                "short_name": "aCCF-H2O"
            })
        return accf


    def accf_o3(self,
                geopotential: np.ndarray | xr.DataArray,
                temperature: np.ndarray | xr.DataArray) -> np.ndarray | xr.DataArray:
        """
        Determines the O3 (ozone) aCCF.
        :param geopotential: [m2/s2]
        :param temperature: [K]
        :return: aCCF-O3 in [K / kg(NO2)]
        """
        beta0, beta1, beta2, beta3 = self.coeffs["o3"]

        accf = beta0 + (beta1 * temperature) + (beta2 * geopotential) + (beta3 * temperature * geopotential)
        accf = accf / ACCFParams.scaling_factor[self.params["version"]]["O3"]
        accf = accf * ACCFParams.efficacy_factors[self.params["efficacy"]]["O3"]
        accf = accf.clip(min=0)

        if isinstance(accf, xr.DataArray):
            return accf.assign_attrs({
                "units": "K kg(NO2)**-1",
                "long_name": "algorithmic climate change function of ozone",
                "short_name": "aCCF-O3"
            })
        return accf

    def _f_in(self, n, phi, timestamp):
        """
        Calculates solar irradiance F_in.
        :param n: day of the year (begins with 0)
        :param phi: latitude
        :param timestamp: pd.DateTime format
        :return: F_in in [W / m2]
        """
        delta = -23.44 * np.cos(np.deg2rad(360 / 365 * (n + 10)))

        sin_phi = np.sin(np.deg2rad(phi))
        cos_phi = np.cos(np.deg2rad(phi))
        sin_delta = np.sin(np.deg2rad(delta))
        cos_delta = np.cos(np.deg2rad(delta))

        if isinstance(phi, xr.DataArray):
            cos_theta = np.outer(sin_phi, sin_delta) + np.outer(cos_phi, cos_delta)
            f = 1360 * cos_theta
            return xr.DataArray(
                f,
                dims=["latitude", "time"],
                coords={"latitude": phi, "time": timestamp}
            )
        else:
            cos_theta = sin_phi * sin_delta + cos_phi * cos_delta
            f = 1360 * cos_theta
            return f

    def accf_ch4(
            self,
            geopotential: np.ndarray | xr.DataArray, timestamp: pd.DataFrame, latitude: np.ndarray,
            include_pmo: bool = False
    ) -> np.ndarray:
        """
        Determines the CH4 (methane) aCCF.
        :param geopotential: [m2/s2]
        :param timestamp: pd.DateTime format
        :param latitude: -90 to 90
        :param include_pmo: boolean
        :return: aCCF-CH4 in [K / kg(NO2)]
        """
        beta0, beta1, beta2, beta3 = self.coeffs["ch4"]

        # N begins with 0
        if not isinstance(timestamp, xr.DataArray):
            days = pd.DatetimeIndex(timestamp).dayofyear.values - 1
        else:
            # 2. Grid Mode (xarray MetDataset)
            days = timestamp.dt.dayofyear - 1
        f_in = self._f_in(n=days, phi=latitude, timestamp=timestamp)

        accf = beta0 + (beta1 * geopotential) + (beta2 * f_in) + (beta3 * geopotential * f_in)
        accf = accf / ACCFParams.scaling_factor[self.params["version"]]["CH4"]
        accf = accf * ACCFParams.efficacy_factors[self.params["efficacy"]]["CH4"]
        accf = accf.clip(max=0)

        if include_pmo:
            accf *= 1.29

        if isinstance(accf, xr.DataArray):
            return accf.assign_attrs({
                "units": "K kg(NO2)**-1",
                "long_name": "algorithmic climate change function of methane",
                "short_name": "aCCF-CH4"
            })
        return accf

    @overload
    def eval(self, source: Flight, **params: Any) -> Flight: ...

    @overload
    def eval(self, source: GeoVectorDataset, **params: Any) -> GeoVectorDataset: ...

    @overload
    def eval(self, source: MetDataset | None = ..., **params: Any) -> MetDataset: ...

    def eval(
        self, source: GeoVectorDataset | Flight | MetDataset | None = None, **params: Any
    ) -> GeoVectorDataset | Flight | MetDataset:
        """Evaluate ACCFs along trajectory or on meteorology grid."""
        self.update_params(params)
        if source is None:
            source = self.met
        self.set_source(source)

        # 1. Grid Mode
        if isinstance(self.source, MetDataset) or self.source is None:
            ds = (self.met if self.source is None else self.source).data
            # aCCF validity
            ds = ds.sel(
                latitude=slice(self.params["lat_min"], self.params["lat_max"]),
                longitude=slice(self.params["lon_min"], self.params["lon_max"]),
                level=slice(self.params["air_pressure_min"], self.params["air_pressure_max"])
            )
            warnings.warn(
                f"""
                Note that the following validity bounds were applied:
                latitude: {self.params["lat_min"], self.params["lat_max"]}
                longitude: {self.params["lon_min"], self.params["lon_max"]}
                level: {self.params["air_pressure_min"], self.params["air_pressure_max"]}
                """
            )
            ds["aCCF_H2O"] = self.accf_h2o(ds["potential_vorticity"])
            ds["aCCF_O3"] = self.accf_o3(ds["geopotential"], ds["air_temperature"])
            ds["aCCF_CH4"] = self.accf_ch4(ds["geopotential"], ds["time"], ds["latitude"], include_pmo=self.params["include_pmo"])
            ds["aCCF_NOx"] = (ds["aCCF_O3"] + ds["aCCF_CH4"]).assign_attrs({
                "units": "K kg(NO2)**-1",
                "long_name": "algorithmic climate change function of nitrogen oxides",
                "short_name": "aCCF-NOx"
            })
            
            result = MetDataset(ds)
            result.data.attrs["pycontrails_version"] = pycontrails.__version__
            return result

        # 2. Vector / Trajectory Mode
        if isinstance(self.source, GeoVectorDataset):
            self.downselect_met()

            # Ensure met fields are intersected onto the flight path in one quick loop
            for name in ["potential_vorticity", "geopotential", "air_temperature"]:
                if name not in self.source:
                    self.source[name] = self.source.intersect_met(self.met[name])

            # Evaluate the aCCFs along the trajectory
            self.source["aCCF_H2O"] = self.accf_h2o(self.source["potential_vorticity"])
            self.source["aCCF_O3"] = self.accf_o3(self.source["geopotential"], self.source["air_temperature"])
            self.source["aCCF_CH4"] = self.accf_ch4(
                self.source["geopotential"], self.source["time"], self.source["latitude"], include_pmo=self.params["include_pmo"]
            )
            self.source["aCCF_NOx"] = self.source["aCCF_O3"] + self.source["aCCF_CH4"]

            # Aggregate the climate effect to attrs
            # aCCF validity
            self.source["in_horizontal_bounds"] = (
                self.source.dataframe["latitude"].between(self.params["lat_min"], self.params["lat_max"])
                & self.source.dataframe["longitude"].between(self.params["lon_min"], self.params["lon_max"])
            )
            self.source["in_vertical_bounds"] = self.source.dataframe["air_pressure"].between(
                self.params["air_pressure_min"] * 100, self.params["air_pressure_max"] * 100
            ) # air_pressure on source is given in [Pa]

            accf_valid = self.source.dataframe[(self.source.dataframe["in_horizontal_bounds"]) & (self.source.dataframe["in_vertical_bounds"])]

            n_segments = self.source.dataframe.shape[0]
            n_segments_valid = accf_valid.shape[0]
            if n_segments != n_segments_valid:
                valid_share = n_segments_valid / n_segments * 100
                warnings.warn(
                    f"""
                    Note that only {valid_share:.1f}% of all cruise flight segments
                    could be evaluated for their climate effect using the aCCFs due to their limited validity.
                    """
                )

            # Calculate aggregated climate effect
            for name in ["nox_ei", "fuel_burn"]:
                if name not in self.source:
                    raise ValueError(f"{name} is not provided on source")
            self.source.attrs["h2o_pulse_atr20"] = np.nansum(accf_valid["aCCF_H2O"] * accf_valid["fuel_burn"]) * ACCFParams.efficacy_factors[self.params["efficacy"]]["H2O"]
            self.source.attrs["o3_pulse_atr20"] = np.nansum(accf_valid["aCCF_O3"] * accf_valid["nox_ei"] * accf_valid["fuel_burn"]) * ACCFParams.efficacy_factors[self.params["efficacy"]]["O3"]
            self.source.attrs["ch4_pulse_atr20"] = np.nansum(accf_valid["aCCF_CH4"] * accf_valid["nox_ei"] * accf_valid["fuel_burn"]) * ACCFParams.efficacy_factors[self.params["efficacy"]]["CH4"]
            self.source.attrs["nox_pulse_atr20"] = self.source.attrs["o3_pulse_atr20"] + self.source.attrs["ch4_pulse_atr20"]

            self.source.attrs["pycontrails_version"] = pycontrails.__version__
            return self.source

        raise NotImplementedError(f"Unsupported source type: {type(self.source)}")
