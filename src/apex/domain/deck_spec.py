"""A synthetic case described in typed terms, for the case-converter to create.

A :class:`DeckSpec` is what a request in words becomes: a box grid, its property
statistics, wells with their controls, a schedule of report steps and a fluid preset. A
deck skill writes the deck from it; the truth and the prior are drawn from the property
statistics. Everything is validated here, before a deck is written.

Fluid presets:

- ``oil_water`` (:class:`FluidSpec`): dead oil and water, producers with water injectors;
- ``co2_brine`` (:class:`Co2BrineSpec`): CO2 storage in a saline aquifer, OPM Flow's
  ``CO2STORE`` (CO2 and brine properties computed by the simulator from the pressure,
  temperature and salinity; CO2 dissolving in the brine), CO2 injectors with optional brine
  producers. A CO2 rate is in sm3/day of CO2 at surface conditions: 1 Mt a year is about
  1.46e6 sm3/day (CO2's surface density, 1.87 kg/m3).

Units are METRIC: metres, bar, mD, sm3/day, days, degrees Celsius.
"""

import datetime as dt
import math
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: At most this many cells: a synthetic case runs on a workstation.
MAX_CELLS = 1_000_000


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GridSpec(_Spec):
    """A box of ``nx x ny x nz`` cells of ``dx x dy x dz`` metres, its top at ``top``."""

    nx: int = Field(ge=1, le=1000)
    ny: int = Field(ge=1, le=1000)
    nz: int = Field(ge=1, le=200)
    dx: float = Field(default=50.0, gt=0)
    dy: float = Field(default=50.0, gt=0)
    dz: float = Field(default=5.0, gt=0)
    top: float = Field(default=2000.0, gt=0)

    @model_validator(mode="after")
    def _small_enough(self) -> Self:
        if self.nx * self.ny * self.nz > MAX_CELLS:
            raise ValueError(f"more than {MAX_CELLS} cells")
        return self


class WellSpec(_Spec):
    """A vertical well: its column (1-based), completed layers and control."""

    name: str = Field(pattern=r"^[A-Z][A-Z0-9_-]{0,7}$")
    kind: Literal["producer", "injector"]
    i: int = Field(ge=1)
    j: int = Field(ge=1)
    k_top: int = Field(default=1, ge=1)
    k_bottom: int | None = Field(default=None, ge=1, description="Default: the last layer.")
    control: Literal["bhp", "liquid_rate", "water_rate", "gas_rate"] = Field(
        description="Producers: bhp or liquid_rate; injectors: water_rate (oil_water), "
        "gas_rate (co2_brine: CO2) or bhp."
    )
    target: float = Field(gt=0, description="The rate (sm3/day) or the pressure (bar).")
    bhp_limit: float = Field(gt=0, description="The pressure limit (bar): minimum or maximum.")

    @model_validator(mode="after")
    def _controls(self) -> Self:
        allowed = {"producer": ("bhp", "liquid_rate"),
                   "injector": ("bhp", "water_rate", "gas_rate")}  # fmt: skip
        if self.control not in allowed[self.kind]:
            raise ValueError(f"a {self.kind} is controlled by {' or '.join(allowed[self.kind])}")
        return self


class ScheduleSpec(_Spec):
    """Report steps of ``step_days`` from ``start``, ``steps`` of them."""

    start: dt.date = dt.date(2025, 1, 1)
    step_days: int = Field(default=30, ge=1, le=3650)
    steps: int = Field(default=24, ge=1, le=600)

    def report_dates(self) -> list[dt.date]:
        """Return the report dates, after ``start``."""
        return [self.start + dt.timedelta(days=self.step_days * (k + 1)) for k in range(self.steps)]


class FluidSpec(_Spec):
    """The ``oil_water`` preset: dead oil and water, Corey relative permeabilities."""

    preset: Literal["oil_water"] = "oil_water"
    initial_pressure: float = Field(default=200.0, gt=1, description="bar, at the top")
    oil_viscosity: float = Field(default=1.0, gt=0, description="cP")
    water_viscosity: float = Field(default=0.5, gt=0, description="cP")
    oil_density: float = Field(default=850.0, gt=0, description="kg/m3")
    water_density: float = Field(default=1000.0, gt=0, description="kg/m3")
    swc: float = Field(default=0.2, ge=0, lt=1, description="Connate water saturation.")
    sorw: float = Field(default=0.2, ge=0, lt=1, description="Residual oil to water.")
    corey_water: float = Field(default=2.0, gt=0)
    corey_oil: float = Field(default=2.0, gt=0)
    krw_max: float = Field(default=0.6, gt=0, le=1)
    kro_max: float = Field(default=0.9, gt=0, le=1)
    rock_compressibility: float = Field(default=4.0e-5, ge=0, description="1/bar")

    @model_validator(mode="after")
    def _mobile(self) -> Self:
        if self.swc + self.sorw >= 1:
            raise ValueError("swc + sorw must leave mobile oil (< 1)")
        return self


class Co2BrineSpec(_Spec):
    """The ``co2_brine`` preset: CO2 into brine (OPM Flow's CO2STORE), Corey drainage curves.

    The aquifer is all brine at the start (hydrostatic, from ``initial_pressure`` at the
    top), at one temperature. Brine and CO2 densities, viscosities and the CO2's solubility
    come from the simulator's correlations; only the salinity and temperature are given.
    """

    preset: Literal["co2_brine"]
    initial_pressure: float = Field(default=200.0, gt=1, description="bar, at the top")
    temperature: float = Field(default=60.0, gt=0, le=200, description="degrees C, uniform")
    salinity: float = Field(default=0.62, ge=0, le=6.0, description="mol NaCl per kg water")
    swc: float = Field(default=0.2, ge=0, lt=1, description="Irreducible water saturation.")
    sgc: float = Field(default=0.05, ge=0, lt=1, description="Critical gas (CO2) saturation.")
    corey_gas: float = Field(default=2.0, gt=0)
    corey_water: float = Field(default=3.0, gt=0)
    krg_max: float = Field(default=1.0, gt=0, le=1, description="CO2's at irreducible water.")
    krw_max: float = Field(default=1.0, gt=0, le=1)
    rock_compressibility: float = Field(default=4.5e-5, ge=0, description="1/bar")

    @model_validator(mode="after")
    def _mobile(self) -> Self:
        if self.swc + self.sgc >= 1:
            raise ValueError("swc + sgc must leave mobile CO2 (< 1)")
        return self


class PropertySpec(_Spec):
    """The truth's and the prior's statistics: ln PERMX and PORO as Gaussian fields."""

    permx_md: float = Field(default=100.0, gt=0, description="The geometric mean, mD.")
    permx_log_std: float = Field(default=1.0, gt=0, description="Std of ln(PERMX).")
    poro: float = Field(default=0.2, gt=0, lt=1)
    poro_std: float = Field(default=0.03, gt=0)
    range_cells: float = Field(default=8.0, gt=0, description="The variogram's range, cells.")
    kv_kh: float = Field(default=0.1, gt=0, le=1, description="PERMZ / PERMX.")

    @property
    def log_permx(self) -> float:
        """The mean of ln(PERMX)."""
        return math.log(self.permx_md)


class DeckSpec(_Spec):
    """A synthetic case to create (see the module)."""

    name: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    grid: GridSpec
    wells: list[WellSpec] = Field(min_length=1)
    schedule: ScheduleSpec = Field(default_factory=ScheduleSpec)
    fluid: FluidSpec | Co2BrineSpec = Field(default_factory=FluidSpec, discriminator="preset")
    properties: PropertySpec = Field(default_factory=PropertySpec)

    @model_validator(mode="after")
    def _wells_in_the_grid(self) -> Self:
        names = [w.name for w in self.wells]
        if len(set(names)) != len(names):
            raise ValueError("well names must differ")
        co2 = self.fluid.preset == "co2_brine"
        if co2 and not any(w.kind == "injector" for w in self.wells):
            raise ValueError("at least one CO2 injector")
        if not co2 and not any(w.kind == "producer" for w in self.wells):
            raise ValueError("at least one producer")
        for well in self.wells:
            if co2 and well.control == "water_rate":
                raise ValueError(f"{well.name}: co2_brine injectors inject CO2 (gas_rate or bhp)")
            if not co2 and well.control == "gas_rate":
                raise ValueError(f"{well.name}: oil_water injectors inject water (water_rate "
                                 "or bhp); gas_rate is co2_brine's")  # fmt: skip
        pressure = self.fluid.initial_pressure
        for well in self.wells:
            limit = well.target if well.control == "bhp" else well.bhp_limit
            if well.kind == "producer" and not limit < pressure:
                raise ValueError(
                    f"{well.name}'s bottom-hole pressure {limit:g} bar is not below the initial "
                    f"pressure {pressure:g}: it would not produce"
                )
            if well.kind == "injector" and not limit > pressure:
                raise ValueError(
                    f"{well.name}'s bottom-hole pressure {limit:g} bar is not above the initial "
                    f"pressure {pressure:g}: it would not inject"
                )
            bottom = well.k_bottom or self.grid.nz
            if not (well.i <= self.grid.nx and well.j <= self.grid.ny and
                    well.k_top <= bottom <= self.grid.nz):  # fmt: skip
                raise ValueError(f"{well.name} is outside the {self.grid.nx} x {self.grid.ny} x "
                                 f"{self.grid.nz} grid")  # fmt: skip
        return self


def five_spot(name: str = "five_spot", n: int = 21, layers: int = 3) -> DeckSpec:
    """Return the rules' case: four producers in the corners and an injector in the middle.

    Producers on liquid rate with a 100 bar minimum, the injector on water rate with a
    400 bar maximum, monthly reports for two years (the 5Spots example's shape).
    """
    corner = 2
    far = n - 1
    middle = (n + 1) // 2
    producers = [
        WellSpec(name=f"PROD{k + 1}", kind="producer", i=i, j=j, control="liquid_rate",
                 target=100.0, bhp_limit=100.0)
        for k, (i, j) in enumerate([(corner, corner), (far, corner), (corner, far), (far, far)])
    ]  # fmt: skip
    injector = WellSpec(name="INJ", kind="injector", i=middle, j=middle, control="water_rate",
                        target=400.0, bhp_limit=400.0)  # fmt: skip
    return DeckSpec(name=name, grid=GridSpec(nx=n, ny=n, nz=layers),
                    wells=[*producers, injector])  # fmt: skip


def co2_injection(name: str = "co2_box", n: int = 21, layers: int = 4) -> DeckSpec:
    """Return a CO2 storage case: one CO2 injector in the middle, brine producers at corners.

    About 0.5 Mt of CO2 a year (7.3e5 sm3/day) at the base, within 400 bar; two brine
    producers relieve the pressure; monthly reports for three years.
    """
    middle = (n + 1) // 2
    injector = WellSpec(name="INJ", kind="injector", i=middle, j=middle, k_top=layers,
                        control="gas_rate", target=7.3e5, bhp_limit=400.0)  # fmt: skip
    producers = [
        WellSpec(name=f"PROD{k + 1}", kind="producer", i=i, j=j, control="liquid_rate",
                 target=500.0, bhp_limit=100.0)
        for k, (i, j) in enumerate([(2, 2), (n - 1, n - 1)])
    ]  # fmt: skip
    return DeckSpec(name=name, grid=GridSpec(nx=n, ny=n, nz=layers),
                    wells=[injector, *producers], schedule=ScheduleSpec(steps=36),
                    fluid=Co2BrineSpec(preset="co2_brine"))  # fmt: skip
