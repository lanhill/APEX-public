<%!
import numpy as np
%>
-- The 3dBox AVO case (the APEX authors' own, set up for PET; GPL-3.0-or-later): its PET
-- deck, RUNFILE.mako, for APEX: a 60 x 60 x 5 box of 30 m cells from 2355 m, its grid given
-- by DX/DY/DZ/TOPS (its Grid.grdecl holds the same box as COORD/ZCORN). The state is the
-- log-permeability and, when the state holds it, the porosity (otherwise the true porosity,
-- poro.inc). permx_cap (template_values; 6, the PET deck's, by default) caps ln PERMX.
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

-- A restart at every report time, so at every vintage.
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

TSTEP
10*400 /

END
