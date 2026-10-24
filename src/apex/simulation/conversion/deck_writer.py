"""Write an Eclipse-format deck template from a :class:`~apex.domain.deck_spec.DeckSpec`.

The ``oil_water`` preset (dead oil and water, METRIC) that OPM Flow and Eclipse 100 both
run: a box grid (``DX``/``DY``/``DZ``/``TOPS``), ln PERMX and PORO from the member's state
(``permx``, ``poro``; PERMY = PERMX, PERMZ = ``kv_kh`` PERMX), Corey ``SWOF``, ``PVDO``,
``PVTW``, ``DENSITY``, ``ROCK``, equilibration with the oil-water contact below the box
(all oil at connate water), wells on rate or pressure with their limits, and a report every
``step_days`` (``TSTEP``). The summary has the field's and every well's rates and
pressures, which the converter observes.
"""

from apex.domain.deck_spec import DeckSpec, FluidSpec, WellSpec

#: Rows of the SWOF table between connate water and residual oil.
SWOF_ROWS = 10
#: Eclipse's month names (not the locale's).
MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
#: The dead oil's formation volume factor: 1.05 at 1 bar, falling linearly to 1.01 at the
#: table's top pressure (twice the highest pressure of the case, at least 400 bar).
PVDO_BO = (1.05, 1.01)
PVDO_POINTS = 4
#: The water's formation volume factor and compressibility (1/bar) at the reference.
WATER_FVF, WATER_COMPRESSIBILITY = 1.01, 4.0e-5
#: Well connections' diameter (m).
WELL_DIAMETER = 0.2

#: The summary vectors written per well; the converter observes them.
WELL_VECTORS = {"producer": ("WOPR", "WWPR", "WBHP"), "injector": ("WWIR", "WBHP")}


def swof(fluid: FluidSpec) -> list[tuple[float, float, float, float]]:
    """Return the Corey SWOF rows ``(sw, krw, krow, pcow)``, from connate water to 1."""
    mobile = 1.0 - fluid.swc - fluid.sorw
    rows = []
    for k in range(SWOF_ROWS + 1):
        sw = fluid.swc + mobile * k / SWOF_ROWS
        s = (sw - fluid.swc) / mobile
        rows.append((sw, fluid.krw_max * s**fluid.corey_water,
                     fluid.kro_max * (1.0 - s) ** fluid.corey_oil, 0.0))  # fmt: skip
    if fluid.sorw > 1e-6:  # beyond residual oil: water only, its permeability rising to 1
        rows.append((1.0, 1.0, 0.0, 0.0))
    return rows


def pvdo(spec: DeckSpec) -> list[tuple[float, float]]:
    """Return the dead oil's ``(pressure, Bo)`` rows, covering every pressure of the case."""
    pressures = [spec.fluid.initial_pressure, *(w.bhp_limit for w in spec.wells),
                 *(w.target for w in spec.wells if w.control == "bhp")]  # fmt: skip
    top = max(400.0, 2.0 * max(pressures))
    rows = []
    for k in range(PVDO_POINTS):
        p = 1.0 + (top - 1.0) * k / (PVDO_POINTS - 1)
        rows.append((p, PVDO_BO[0] + (PVDO_BO[1] - PVDO_BO[0]) * k / (PVDO_POINTS - 1)))
    return rows


def write_deck(spec: DeckSpec) -> str:
    """Return the deck template of ``spec`` (see the module)."""
    grid, fluid, props = spec.grid, spec.fluid, spec.properties
    cells = grid.nx * grid.ny * grid.nz
    bottom = grid.top + grid.nz * grid.dz
    lines = [
        "<%!\nimport math\n%>",
        f"-- {spec.name}: a synthetic case created by APEX's case-converter (oil_water).",
        "RUNSPEC", "TITLE", f" {spec.name}", "DIMENS", f" {grid.nx} {grid.ny} {grid.nz} /",
        "OIL", "WATER", "METRIC", "START",
        f" {spec.schedule.start.day} '{MONTHS[spec.schedule.start.month - 1]}' "
        f"{spec.schedule.start.year} /",
        "WELLDIMS", f" {len(spec.wells)} {grid.nz} 1 {len(spec.wells)} /",
        "TABDIMS", f" 1 1 {SWOF_ROWS + 2} {PVDO_POINTS} /",
        "GRID",
        "DX", f" {cells}*{grid.dx!r} /", "DY", f" {cells}*{grid.dy!r} /",
        "DZ", f" {cells}*{grid.dz!r} /", "TOPS", f" {grid.nx * grid.ny}*{grid.top!r} /",
        "-- PERMX: the member's state variable permx (its logarithm), written by APEX",
        "PERMX", "% for value in permx:", '${"%.10g" % math.exp(value)}', "% endfor", "/",
        "-- PORO: the member's state variable poro, written by APEX",
        "PORO", "% for value in poro:", '${"%.10g" % value}', "% endfor", "/",
        "COPY", " 'PERMX' 'PERMY' /", " 'PERMX' 'PERMZ' /", "/",
        "MULTIPLY", f" 'PERMZ' {props.kv_kh!r} /", "/",
        "PROPS",
        "SWOF", *(f" {sw:.6f} {krw:.6f} {kro:.6f} {pc:g}" for sw, krw, kro, pc in swof(fluid)),
        "/",
        "PVDO", *(f" {p:.6g} {b:.6g} {fluid.oil_viscosity:g}" for p, b in pvdo(spec)), "/",
        "PVTW", f" {fluid.initial_pressure:g} {WATER_FVF:g} {WATER_COMPRESSIBILITY:g} "
        f"{fluid.water_viscosity:g} 0 /",
        "DENSITY", f" {fluid.oil_density:g} {fluid.water_density:g} 1.0 /",
        "ROCK", f" {fluid.initial_pressure:g} {fluid.rock_compressibility:g} /",
        "SOLUTION",
        # The oil-water contact 100 m below the box: all oil, at connate water.
        "EQUIL", f" {grid.top:g} {fluid.initial_pressure:g} {bottom + 100.0:g} 0 {grid.top:g} 0 /",
        "SUMMARY", "FOPR", "FWPR", "FWIR", "FPR",
        *_summary(spec.wells),
        "SCHEDULE",
        "WELSPECS",
        *(f" '{w.name}' 'G1' {w.i} {w.j} 1* '{'OIL' if w.kind == 'producer' else 'WATER'}' /"
          for w in spec.wells),
        "/",
        "COMPDAT",
        *(f" '{w.name}' {w.i} {w.j} {w.k_top} {w.k_bottom or grid.nz} 'OPEN' 2* {WELL_DIAMETER} /"
          for w in spec.wells),
        "/",
        *_controls(spec.wells),
        "TSTEP", f" {spec.schedule.steps}*{spec.schedule.step_days} /",
        "END",
    ]  # fmt: skip
    return "\n".join(lines) + "\n"


def _summary(wells: list[WellSpec]) -> list[str]:
    lines: list[str] = []
    for vector in ("WOPR", "WWPR", "WWIR", "WBHP", "WWCT"):
        lines += [vector, *(f" '{w.name}'" for w in wells), "/"]
    return lines


def _controls(wells: list[WellSpec]) -> list[str]:
    producers = [w for w in wells if w.kind == "producer"]
    injectors = [w for w in wells if w.kind == "injector"]
    lines: list[str] = []
    if producers:
        lines.append("WCONPROD")
        for w in producers:
            # WCONPROD: well status mode ORAT WRAT GRAT LRAT RESV BHP.
            if w.control == "liquid_rate":
                lines.append(f" '{w.name}' 'OPEN' 'LRAT' 3* {w.target:g} 1* {w.bhp_limit:g} /")
            else:
                lines.append(f" '{w.name}' 'OPEN' 'BHP' 5* {w.target:g} /")
        lines.append("/")
    if injectors:
        lines.append("WCONINJE")
        for w in injectors:
            # WCONINJE: well type status mode RATE RESV BHP.
            if w.control == "water_rate":
                lines.append(f" '{w.name}' 'WATER' 'OPEN' 'RATE' {w.target:g} 1* "
                             f"{w.bhp_limit:g} /")  # fmt: skip
            else:
                lines.append(f" '{w.name}' 'WATER' 'OPEN' 'BHP' 2* {w.target:g} /")
        lines.append("/")
    return lines
