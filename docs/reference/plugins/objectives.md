---
audience: public
generated: reference
---

# Objectives: reference

Every one of these 15 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `ackley`

Ackley's function; minimum 0 at 0, a nearly flat outer region with many local minima.

Select it with:

```toml
[objective]
name = "ackley"
```


## `bnh`

BNH: `f1 = 4 x1^2 + 4 x2^2`, `f2 = (x1-5)^2 + (x2-5)^2` with two constraints.

Select it with:

```toml
[objective]
name = "bnh"
```


## `dtlz1`

DTLZ1: a linear front, the simplex `sum f = 0.5`, with `11^k - 1` local fronts.

Select it with:

```toml
[objective]
name = "dtlz1"
```

| option | type | default | description |
|---|---|---|---|
| `n_obj` | integer | 3 | Number of objectives, at least 2 (default 3). The controls must number at least `n_obj`: the first `n_obj - 1` place a point on the front and the rest set its distance from it. |

## `dtlz2`

DTLZ2: the unit sphere's positive part (`n_x` = 12 by default for 3 objectives).

Select it with:

```toml
[objective]
name = "dtlz2"
```

| option | type | default | description |
|---|---|---|---|
| `n_obj` | integer | 3 | Number of objectives, at least 2 (default 3). The controls must number at least `n_obj`: the first `n_obj - 1` place a point on the front and the rest set its distance from it. |

## `fon`

FON (n = 3): `f1,2 = 1 - exp(-sum (x_i -+ 1/sqrt(3))^2)` on `[-4, 4]`; non-convex.

Select it with:

```toml
[objective]
name = "fon"
```


## `kur`

KUR (n = 3) on `[-5, 5]`: a disconnected front (pymoo's form).

Select it with:

```toml
[objective]
name = "kur"
```


## `quadratic`

PET's quadratic `f(x) = (x - 1)^T A (x - 1)`, `A = I / 2`; minimum 0 at `x = 1`.

Select it with:

```toml
[objective]
name = "quadratic"
```


## `rastrigin`

Rastrigin's function `10 n + sum x_i^2 - 10 cos(2 pi x_i)`; minimum 0 at 0.

Select it with:

```toml
[objective]
name = "rastrigin"
```


## `rosenbrock`

SciPy's Rosenbrock function `sum 100 (x_i+1 - x_i^2)^2 + (1 - x_i)^2`; minimum 0 at 1.

Select it with:

```toml
[objective]
name = "rosenbrock"
```


## `sch`

SCH: `f1 = x^2`, `f2 = (x - 2)^2` on `[-1000, 1000]`; front for `x` in [0, 2].

Select it with:

```toml
[objective]
name = "sch"
```


## `srn`

SRN: two quadratic objectives on `[-20, 20]^2` with two constraints.

Select it with:

```toml
[objective]
name = "srn"
```


## `tnk`

TNK: `f = (x1, x2)` on `[0, pi]^2` with two constraints; a disconnected front.

Select it with:

```toml
[objective]
name = "tnk"
```


## `zdt1`

ZDT1: `f2 = g (1 - sqrt(f1 / g))`; convex front `f2 = 1 - sqrt(f1)`.

Select it with:

```toml
[objective]
name = "zdt1"
```


## `zdt2`

ZDT2: `f2 = g (1 - (f1 / g)^2)`; non-convex front `f2 = 1 - f1^2`.

Select it with:

```toml
[objective]
name = "zdt2"
```


## `zdt3`

ZDT3: `f2 = g (1 - sqrt(f1/g) - f1/g sin(10 pi f1))`; five disconnected pieces.

Select it with:

```toml
[objective]
name = "zdt3"
```
