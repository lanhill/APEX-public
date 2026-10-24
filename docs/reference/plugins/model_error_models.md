---
audience: public
generated: reference
---

# Model error models: reference

Every one of these 1 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `pg_lstm`

Physics-guided LSTM well residual models, one per well.

Select it with:

```toml
[model_error]
name = "pg_lstm"
```

| option | type | default | description |
|---|---|---|---|
| `hidden_size` | integer | 20 | |
| `initial_schedule_file` | string or string or null | | |
| `injector_data` | array | ["WBHP"] | |
| `injectors` | array | | |
| `phases` | array | ["oil", "water"] | |
| `producer_data` | array | ["WOPR", "WBHP", "WWCT"] | |
| `producers` | array | | |
| `report_steps` | array | required | |
| `schedule_file` | string or string | required | |
| `shut_wells` | string | zero | |
| `state_variable` | string | nn_params | |
