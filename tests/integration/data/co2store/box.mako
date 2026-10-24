<%!
import numpy as np
%>
-- The CO2 box (APEX's own; tests/integration/test_geophysics_co2_box.py): the
-- 3dBox AVO case's grid (60 x 60 x 5
-- cells of 30 m from 2355 m) as an OPM CO2STORE gas-water deck. Layer 1 is a tight caprock
-- (0.001 mD, porosity 0.05); layers 2-5 are the reservoir, whose log-permeability is the
-- state (14,400 values, the 3dBox truth's layers 2-5 for the truth), capped at 8 (2981 mD;
-- the 3dBox deck's cap of 6 would cut 15 % of this truth, up to 6.87). One CO2 injector at the
-- base of the centre column injects for 5 years. The reservoir's edge columns have 1000 times
-- their pore volume, an open boundary: a closed 1.8 km box would pressure up far beyond a
-- real store. One reservoir temperature, 40 C (RTEMPVD constant; the PEM's temperature_c):
-- colder than the depth implies (a normal gradient gives 75-80 C), chosen to keep CO2 close
-- to its critical region (dense, 850-880 kg/m3; T/Tc 1.03), where Span & Wagner and Batzle &
-- Wang's extrapolated correlation differ most, which configuration (b) is about.
RUNSPEC
TITLE
 APEX CO2 BOX
DIMENS
 60 60 5 /
EQLDIMS
/
TABDIMS
/
WATER
GAS
CO2STORE
DISGASW
METRIC
START
 1 'JAN' 2020 /
WELLDIMS
 1 5 1 1 /
UNIFOUT
GRID
INIT
DX
 18000*30 /
DY
 18000*30 /
DZ
 18000*30 /
TOPS
 3600*2355 /
INCLUDE
 '${case_dir}/poro.inc' /
PERMX
 3600*0.001
% for value in permx:
${"%.6g" % np.exp(min(value, 8.0))}
% endfor
/
COPY
 'PERMX' 'PERMY' /
 'PERMX' 'PERMZ' /
/
-- The open boundary: the reservoir layers' edge columns.
BOX
 1 60 1 1 2 5 /
MULTPV
 240*1000 /
BOX
 1 60 60 60 2 5 /
MULTPV
 240*1000 /
BOX
 1 1 2 59 2 5 /
MULTPV
 232*1000 /
BOX
 60 60 2 59 2 5 /
MULTPV
 232*1000 /
ENDBOX
PROPS
ROCK
 240 4.5e-5 /
-- Drainage: critical gas saturation 0.05, irreducible water 0.2, Corey exponents 2 (gas)
-- and 3 (water); no capillary pressure.
SGFN
 0     0     0
 0.05  0     0
 0.2   0.04  0
 0.35  0.16  0
 0.5   0.36  0
 0.65  0.64  0
 0.8   1     0
 1     1     0 /
SWFN
 0.2   0       0
 0.4   0.0156  0
 0.6   0.125   0
 0.8   0.4219  0
 1     1       0 /
-- 0.62 mol/kg NaCl: 36.2 g per kg of water, a mass fraction of 0.0350 (the PEM's salinity).
SALINITY
 0.62 /
SOLUTION
-- Hydrostatic brine from 240 bar at the top; the contacts above the box (all brine).
EQUIL
 2355 240 2000 0 2000 0 /
RTEMPVD
 0    40
 4000 40 /
-- A restart at the start (the seismic baseline) and at every report time.
RPTRST
 BASIC=2 DEN /
SUMMARY
FPR
FGIT
WGIR
 'INJ' /
WBHP
 'INJ' /
SCHEDULE
RPTRST
 BASIC=2 DEN /
WELSPECS
 'INJ' 'G' 30 30 1* 'GAS' /
/
COMPDAT
 'INJ' 30 30 5 5 'OPEN' 1* 1* 0.2 /
/
-- 5e5 sm3/day, about 0.34 Mt/year (OPM's CO2 surface density 1.87 kg/sm3), within 400 bar.
WCONINJE
 'INJ' 'GAS' 'OPEN' 'RATE' 5e5 1* 400 /
/
-- Quarterly report times for 5 years; the seismic monitors are 2022-01-01 and 2025-01-01.
DATES
 1 'APR' 2020 /
 1 'JUL' 2020 /
 1 'OCT' 2020 /
 1 'JAN' 2021 /
 1 'APR' 2021 /
 1 'JUL' 2021 /
 1 'OCT' 2021 /
 1 'JAN' 2022 /
 1 'APR' 2022 /
 1 'JUL' 2022 /
 1 'OCT' 2022 /
 1 'JAN' 2023 /
 1 'APR' 2023 /
 1 'JUL' 2023 /
 1 'OCT' 2023 /
 1 'JAN' 2024 /
 1 'APR' 2024 /
 1 'JUL' 2024 /
 1 'OCT' 2024 /
 1 'JAN' 2025 /
/
END
