<%!
import numpy as np
%>
-- A small box for a prediction with a control scenario: 20 x 20 x 2 cells of 30 m from
-- 2355 m, one water injector and two producers, every well on bottom-hole pressure (the
-- producers at 90 bar, the injector at 300 bar) for the history, three years reported
-- quarterly. The state is the log-permeability (ln mD) of every cell; the porosity is 0.2.
-- Its fluids are the 3dBox case's (pvt.txt; the APEX authors', GPL-3.0-or-later).
-- prepare.py writes deck_forecast.mako from this file: the same history, then the
-- forecast under the producers' bottom-hole pressures of a control scenario.
RUNSPEC

TITLE
 PREDICTION WITH CONTROLS

DIMENS
 20 20 2 /

OIL
WATER
GAS
DISGAS

METRIC

TABDIMS
-- NTSFUN  NTPVT  NSSFUN  NPPVT  NTFIP  NRPVT  NTENDP
     1       1      35      30     5     30      1 /

EQLDIMS
-- NTEQUL  NDRXVD  NDPRVD
   1       5       100 /

WELLDIMS
-- NWMAXZ NCWMAX NGMAXZ MWGMAX
    5      5      2      5 /

START
 01 JAN 2022 /

NOECHO

GRID

DX
 800*30 /
DY
 800*30 /
DZ
 800*15 /
TOPS
 400*2355 /

PORO
 800*0.2 /

PERMX
% for value in permx:
${"%.4f" % np.exp(value)}
% endfor
/

COPY
 'PERMX'  'PERMY'  /
 'PERMX'  'PERMZ' /
/

PROPS    ===============================================================

INCLUDE
 '${case_dir}/pvt.txt' /

SOLUTION ===============================================================

--    DATUM  DATUM   OWC    OWC    GOC    GOC    RSVD   RVVD   SOLN
--    DEPTH  PRESS  DEPTH   PCOW  DEPTH   PCOG  TABLE  TABLE   METH
EQUIL
     2355.00 200.46 3000 0.00  2355.0 0.000     /

SUMMARY ================================================================

RUNSUM

RPTONLY

WWIR
 'INJ1'
/

WOPR
 'PRO1'
 'PRO2'
/

WWPR
 'PRO1'
 'PRO2'
/

SCHEDULE =============================================================

RPTRST
 BASIC=0 /

WELSPECS
 'INJ1' 'G'  1 10 2355 'WATER' 1* 'STD' 1* 1* 1* /
 'PRO1' 'G' 20  1 2355 'OIL'   1* 'STD' 1* 1* 1* /
 'PRO2' 'G' 20 20 2355 'OIL'   1* 'STD' 1* 1* 1* /
/

COMPDAT
 'INJ1'  1 10 1 2 'OPEN' 1* 1* 0.15 1* 5.0 /
 'PRO1' 20  1 1 2 'OPEN' 1* 1* 0.15 1* 5.0 /
 'PRO2' 20 20 1 2 'OPEN' 1* 1* 0.15 1* 5.0 /
/

WCONINJE
 'INJ1' 'WATER' 'OPEN' 'BHP' 2* 300 /
/

WCONPROD
 'PRO1' 'OPEN' 'BHP' 5* 90 /
 'PRO2' 'OPEN' 'BHP' 5* 90 /
/

-- The history: quarterly report times to 2025-01-01.
DATES
 1 'APR' 2022 /
/
DATES
 1 'JUL' 2022 /
/
DATES
 1 'OCT' 2022 /
/
DATES
 1 'JAN' 2023 /
/
DATES
 1 'APR' 2023 /
/
DATES
 1 'JUL' 2023 /
/
DATES
 1 'OCT' 2023 /
/
DATES
 1 'JAN' 2024 /
/
DATES
 1 'APR' 2024 /
/
DATES
 1 'JUL' 2024 /
/
DATES
 1 'OCT' 2024 /
/
DATES
 1 'JAN' 2025 /
/


-- The forecast (written by prepare.py): the producers' bottom-hole pressures (bar)
-- of the control scenario, bhp = [PRO1, PRO2] (the prediction's [[controls.variables]]),
-- from 2025-01-01, reported quarterly for two years.
WCONPROD
 'PRO1' 'OPEN' 'BHP' 5* ${bhp[0]} /
 'PRO2' 'OPEN' 'BHP' 5* ${bhp[1]} /
/
DATES
 1 'APR' 2025 /
/
DATES
 1 'JUL' 2025 /
/
DATES
 1 'OCT' 2025 /
/
DATES
 1 'JAN' 2026 /
/
DATES
 1 'APR' 2026 /
/
DATES
 1 'JUL' 2026 /
/
DATES
 1 'OCT' 2026 /
/
DATES
 1 'JAN' 2027 /
/

END
