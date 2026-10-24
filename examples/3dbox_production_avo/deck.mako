<%!
import numpy as np
%>
-- The 3dBox AVO case (the APEX authors' own, set up for PET; GPL-3.0-or-later): its PET
-- deck, RUNFILE.mako, for APEX: a 60 x 60 x 5 box of 30 m cells from 2355 m, its grid given
-- by DX/DY/DZ/TOPS (its Grid.grdecl holds the same box as COORD/ZCORN). The state is the
-- log-permeability and, when the state holds it, the porosity (otherwise the true porosity,
-- poro.inc). permx_cap (template_values; 6, the PET deck's, by default) caps ln PERMX. It
-- reports monthly, with restarts only at the AVO vintages; deck_pet.mako is the PET deck's
-- schedule (10 steps of 400 days, a restart at each), for the SimulatorWrap comparisons.
<%
    poro_state = context.get("poro", None)
    permx_cap = float(context.get("permx_cap", 6.0))
%>
RUNSPEC

TITLE
 3DBOX AVO MODEL

DIMENS
 60 60 5 /

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
    15     15     2      20 /

START
 01 JAN 2022 /

NSTACK
 25 /

NOECHO

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

% if poro_state is None:
INCLUDE
 '${case_dir}/poro.inc' /
% else:
PORO
% for value in poro_state:
${"%.4f" % value}
% endfor
/
% endif

PERMX
% for value in permx:
${"%.4f" % np.exp(min(value, permx_cap))}
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

RPTSOL
'PRES' 'SWAT' /

-- A restart at the start (PET's baseline, which is the same in every member).
RPTRST
 BASIC=2 DEN RPORV /

SUMMARY ================================================================

RUNSUM

RPTONLY

WWIR
 'INJ1'
 'INJ2'
 'INJ3'
/

WOPR
 'PRO1'
 'PRO2'
 'PRO3'
/

WWPR
 'PRO1'
 'PRO2'
 'PRO3'
/

ELAPSED

SCHEDULE =============================================================

RPTSCHED
 'NEWTON=2' /

-- A restart at the start (the baseline of 4-D and gravity data): OPM Flow writes report
-- step 0's restart by the control in force before the first report time, so BASIC=2 holds
-- until the first DATES (which also gives one at 2022-02-01); then none but at the AVO
-- vintages (set around their dates below).
RPTRST
 BASIC=2 DEN RPORV /

INCLUDE
 '${case_dir}/Schdl.sch' /

WCONINJE
'INJ1' WATER 'OPEN' BHP 2* 300/
'INJ2' WATER 'OPEN' BHP 2* 300/
'INJ3' WATER 'OPEN' BHP 2* 300/
/

WCONPROD
 'PRO1' 'OPEN' BHP 5* 90 /
 'PRO2' 'OPEN' BHP 5* 90 /
 'PRO3' 'OPEN' BHP 5* 90 /
/

-- Monthly report times (the first of every month) and the AVO vintages, with a restart
-- at each vintage only (plain text, so that `apex validate` reads
-- the restarts).
DATES
 1 'FEB' 2022 /
/
RPTRST
 BASIC=0 /
DATES
 1 'MAR' 2022 /
/
DATES
 1 'APR' 2022 /
/
DATES
 1 'MAY' 2022 /
/
DATES
 1 'JUN' 2022 /
/
DATES
 1 'JUL' 2022 /
/
DATES
 1 'AUG' 2022 /
/
DATES
 1 'SEP' 2022 /
/
DATES
 1 'OCT' 2022 /
/
DATES
 1 'NOV' 2022 /
/
DATES
 1 'DEC' 2022 /
/
DATES
 1 'JAN' 2023 /
/
DATES
 1 'FEB' 2023 /
/
DATES
 1 'MAR' 2023 /
/
DATES
 1 'APR' 2023 /
/
DATES
 1 'MAY' 2023 /
/
DATES
 1 'JUN' 2023 /
/
DATES
 1 'JUL' 2023 /
/
DATES
 1 'AUG' 2023 /
/
DATES
 1 'SEP' 2023 /
/
DATES
 1 'OCT' 2023 /
/
DATES
 1 'NOV' 2023 /
/
DATES
 1 'DEC' 2023 /
/
DATES
 1 'JAN' 2024 /
/
DATES
 1 'FEB' 2024 /
/
DATES
 1 'MAR' 2024 /
/
DATES
 1 'APR' 2024 /
/
DATES
 1 'MAY' 2024 /
/
DATES
 1 'JUN' 2024 /
/
DATES
 1 'JUL' 2024 /
/
DATES
 1 'AUG' 2024 /
/
DATES
 1 'SEP' 2024 /
/
DATES
 1 'OCT' 2024 /
/
DATES
 1 'NOV' 2024 /
/
DATES
 1 'DEC' 2024 /
/
DATES
 1 'JAN' 2025 /
/
DATES
 1 'FEB' 2025 /
/
DATES
 1 'MAR' 2025 /
/
DATES
 1 'APR' 2025 /
/
DATES
 1 'MAY' 2025 /
/
DATES
 1 'JUN' 2025 /
/
DATES
 1 'JUL' 2025 /
/
DATES
 1 'AUG' 2025 /
/
DATES
 1 'SEP' 2025 /
/
DATES
 1 'OCT' 2025 /
/
DATES
 1 'NOV' 2025 /
/
DATES
 1 'DEC' 2025 /
/
DATES
 1 'JAN' 2026 /
/
DATES
 1 'FEB' 2026 /
/
DATES
 1 'MAR' 2026 /
/
DATES
 1 'APR' 2026 /
/
DATES
 1 'MAY' 2026 /
/
DATES
 1 'JUN' 2026 /
/
DATES
 1 'JUL' 2026 /
/
DATES
 1 'AUG' 2026 /
/
DATES
 1 'SEP' 2026 /
/
DATES
 1 'OCT' 2026 /
/
DATES
 1 'NOV' 2026 /
/
DATES
 1 'DEC' 2026 /
/
DATES
 1 'JAN' 2027 /
/
DATES
 1 'FEB' 2027 /
/
DATES
 1 'MAR' 2027 /
/
DATES
 1 'APR' 2027 /
/
DATES
 1 'MAY' 2027 /
/
DATES
 1 'JUN' 2027 /
/
RPTRST
 BASIC=2 DEN RPORV /
DATES
 24 'JUN' 2027 /
/
RPTRST
 BASIC=0 /
DATES
 1 'JUL' 2027 /
/
DATES
 1 'AUG' 2027 /
/
DATES
 1 'SEP' 2027 /
/
DATES
 1 'OCT' 2027 /
/
DATES
 1 'NOV' 2027 /
/
DATES
 1 'DEC' 2027 /
/
DATES
 1 'JAN' 2028 /
/
DATES
 1 'FEB' 2028 /
/
DATES
 1 'MAR' 2028 /
/
DATES
 1 'APR' 2028 /
/
DATES
 1 'MAY' 2028 /
/
DATES
 1 'JUN' 2028 /
/
DATES
 1 'JUL' 2028 /
/
DATES
 1 'AUG' 2028 /
/
DATES
 1 'SEP' 2028 /
/
DATES
 1 'OCT' 2028 /
/
DATES
 1 'NOV' 2028 /
/
DATES
 1 'DEC' 2028 /
/
DATES
 1 'JAN' 2029 /
/
DATES
 1 'FEB' 2029 /
/
DATES
 1 'MAR' 2029 /
/
DATES
 1 'APR' 2029 /
/
DATES
 1 'MAY' 2029 /
/
DATES
 1 'JUN' 2029 /
/
DATES
 1 'JUL' 2029 /
/
DATES
 1 'AUG' 2029 /
/
DATES
 1 'SEP' 2029 /
/
DATES
 1 'OCT' 2029 /
/
DATES
 1 'NOV' 2029 /
/
DATES
 1 'DEC' 2029 /
/
DATES
 1 'JAN' 2030 /
/
DATES
 1 'FEB' 2030 /
/
DATES
 1 'MAR' 2030 /
/
DATES
 1 'APR' 2030 /
/
DATES
 1 'MAY' 2030 /
/
DATES
 1 'JUN' 2030 /
/
DATES
 1 'JUL' 2030 /
/
DATES
 1 'AUG' 2030 /
/
DATES
 1 'SEP' 2030 /
/
DATES
 1 'OCT' 2030 /
/
DATES
 1 'NOV' 2030 /
/
DATES
 1 'DEC' 2030 /
/
DATES
 1 'JAN' 2031 /
/
DATES
 1 'FEB' 2031 /
/
DATES
 1 'MAR' 2031 /
/
DATES
 1 'APR' 2031 /
/
DATES
 1 'MAY' 2031 /
/
DATES
 1 'JUN' 2031 /
/
DATES
 1 'JUL' 2031 /
/
DATES
 1 'AUG' 2031 /
/
DATES
 1 'SEP' 2031 /
/
DATES
 1 'OCT' 2031 /
/
DATES
 1 'NOV' 2031 /
/
DATES
 1 'DEC' 2031 /
/
DATES
 1 'JAN' 2032 /
/
DATES
 1 'FEB' 2032 /
/
DATES
 1 'MAR' 2032 /
/
DATES
 1 'APR' 2032 /
/
DATES
 1 'MAY' 2032 /
/
DATES
 1 'JUN' 2032 /
/
DATES
 1 'JUL' 2032 /
/
DATES
 1 'AUG' 2032 /
/
DATES
 1 'SEP' 2032 /
/
DATES
 1 'OCT' 2032 /
/
DATES
 1 'NOV' 2032 /
/
DATES
 1 'DEC' 2032 /
/
RPTRST
 BASIC=2 DEN RPORV /
DATES
 14 'DEC' 2032 /
/
RPTRST
 BASIC=0 /

END
