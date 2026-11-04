# X8 model and initial conditions of the ICINCO 2024 experiments

Used only by the `icinco2024` simulator profile (`fw_jsbgym/envs/sim_profiles.py`).
Do not edit these files. Model changes go to `../x8/x8.xml`.

| File | Exact copy of | SHA-256 |
|------|---------------|---------|
| `x8_icinco2024.xml` | `git show 97e356e^:fw_jsbgym/fdm_descriptions/aircraft/x8/x8.xml` | `e26f40d33742d5ac2fd6930ff2ba582ee401213a9c55ccb9775a0c361d96e01f` |
| `../../../initial_conditions/x8_icinco2024_basic_ic.xml` | `git show 1072782:fw_jsbgym/initial_conditions/x8_basic_ic.xml` | `0ec5bc4a33902dc96af7d0e172109c9b95b9f537e3f36c53a128bc02bcdf0c5a` |

The hashes are checked by `tests/test_sim_profiles.py`.

## Differences from the current files

- Model: drag coefficient table over alpha (-1.57 to 1.57 rad) as before `97e356e` (2025-06-30, "updated Drag model").
- Initial conditions: `ubody` 16.97 m/s at latitude/longitude 0, as before `ad64129` and `76dc60d`
  (2024-12, start position moved to France, `ubody` 16.67 m/s).

## Difference from the model at `1072782`

The model is taken just before `97e356e` because that is the version validated by the ICINCO PPO diagnosis.
It differs from the `1072782` model only by `2ff5b75` (2025-05-14, "gain name typo"): the aileron `pure_gain`
component is named `normalized-aileron-pos` instead of `normalized-elevator-pos`. Only the component name
changed; its output `fcs/effective-aileron-pos-norm` is the same, and no code reads a `fcs/normalized-*-pos`
property.

Evidence: `FW-FlightControl/reproducibility/reports/2026-10-30_icinco_ppo_seed1_drift_diagnosis.md`.

## Author-supplied submission candidate

On 2026-11-03 the author identified `733a2a8bffe9deb7e0f9ca6e202b7da132d1b168`
(2024-06-12) as a likely ICINCO submission snapshot. Its
`jsbgym/initial_conditions/x8_basic_ic.xml` is byte-identical to the profile's
initial-condition file. Its `jsbgym/fdm_descriptions/aircraft/x8/x8.xml`
(SHA-256 `8addc7eac02ea95e469e042e561d15ffec4b426d3dedc1b6804e807a475d2ec0`)
differs from the profile's model only in the component-name correction
described above; its drag table is identical. This is additional source
provenance, not a change to the validated profile assets.
