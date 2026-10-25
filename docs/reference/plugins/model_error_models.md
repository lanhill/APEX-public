---
audience: public
generated: reference
---

# Model error models: reference

This item, with its options, as APEX's code defines it. Written from the code; the user guide teaches their use.

## `pg_lstm`

Physics-guided LSTM well residual models, one per well.

Select it with:

```toml
[model_error]
name = "pg_lstm"
```

| option | type | default | description |
|---|---|---|---|
| `hidden_size` | integer | 20 | LSTM hidden size. |
| `initial_schedule_file` | string or null | | Schedule text read before `schedule_file`: the deck's own SCHEDULE keywords before its included history (e.g. a first COMPDAT), if any. |
| `injector_data` | array | ["WBHP"] | Their data types, e.g. `["WOPR", "WBHP", "WWCT"]` and `["WBHP"]`. |
| `injectors` | array | | Well names. |
| `phases` | array | ["oil", "water"] | The deck's phases (oil required): they set the dynamic features, each phase's density and saturation (`feature_names`). |
| `producer_data` | array | ["WOPR", "WBHP", "WWCT"] | Their data types, e.g. `["WOPR", "WBHP", "WWCT"]` and `["WBHP"]`. |
| `producers` | array | | Well names. |
| `report_steps` | array | required | 1-based report steps of the network's report times (the observation times). |
| `schedule_file` | string | required | The history schedule: one COMPDAT, WCONHIST and WCONINJH block per report step. |
| `shut_wells` | string | zero | `"zero"` (the manuscript): no residual while a well is shut (WBHP excepted); `"ignore"`: residuals at every report time ('s published runs). |
| `state_variable` | string | nn_params | The state variable holding the parameters of every well's network, producers then injectors, each in torch's layout. |
