# Training Results


## Independent Inference
- amortizer_0-time_attention-6_layers: 2.655841112136841
- amortizer_1-time_attention-7_layers: 2.5539121627807617
- amortizer_2-time_attention-8_layers: 2.881528615951538
- amortizer_3-time_attention-9_layers: 2.5146710872650146
- amortizer_4_group_attention-6_layers: 2.9506537914276123
- amortizer_5_group_attention-7_layers: 3.7104172706604004
- amortizer_6_group_attention-8_layers: 3.078146457672119
- amortizer_7_group_attention-9_layers: 2.9382448196411133

Best (3): time attention with 9 invertible layers
(caution with infectiousness of asymptomatic infants, calibration indicates difficulties with identifiability)

### PedCov Only
- amortizer_8-pedcov_only-time_attention-6_layers: 2.4218173027038574
- amortizer_9-pedcov_only-time_attention-7_layers: 2.372084140777588
- amortizer_10-pedcov_only-time_attention-8_layers: 2.2171483039855957
- amortizer_11-pedcov_only-time_attention-9_layers: 2.2801265716552734
- amortizer_12-pedcov_only_group_attention-6_layers: 2.6015894412994385
- amortizer_13-pedcov_only_group_attention-7_layers: 2.6580734252929688
- amortizer_14-pedcov_only_group_attention-8_layers: 2.493386745452881
- amortizer_15-pedcov_only_group_attention-9_layers: 2.8303630352020264

Best (10): time attention with 8 invertible layers 


## Joint Inference

- amortizer_0-time_attention-6_layers: 4.879176139831543
- amortizer_1-time_attention-7_layers: 5.363649845123291
- amortizer_2-time_attention-8_layers: 4.907076358795166
- amortizer_3-time_attention-9_layers: 5.060457706451416
- amortizer_4_group_attention-6_layers: 6.410925388336182
- amortizer_5_group_attention-7_layers: 5.992444038391113
- amortizer_6_group_attention-8_layers: 5.885531902313232
- amortizer_7_group_attention-9_layers: 6.462107181549072

Best (0): time attention with 6 invertible layers (still caution with asymptomatic infants)

### PedCov Only
- amortizer_8-pedcov_only-time_attention-6_layers: 6.209351539611816
- amortizer_9-pedcov_only-time_attention-7_layers: 6.418222904205322
- amortizer_10-pedcov_only-time_attention-8_layers: 5.950998306274414
- amortizer_11-pedcov_only-time_attention-9_layers: 6.147616386413574
- amortizer_12-pedcov_only_group_attention-6_layers: 6.212246894836426
- amortizer_13-pedcov_only_group_attention-7_layers: 6.180913925170898
- amortizer_14-pedcov_only_group_attention-8_layers: 6.2952775955200195
- amortizer_15-pedcov_only_group_attention-9_layers: 6.027039051055908

Best (10): time attention with 8 invertible layers 
