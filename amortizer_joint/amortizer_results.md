
# Both variants jointly and both selection methods
## Loss of networks:
- 0: 12.139  # 6 layers
- 1: 8.3989  # 6 layers, attention (on time points)
- 2: 10.9377 # 7 layers
- 3: 8.439 # 7 layers, attention
- 4: 11.559 # 8 layers
- 5: 8.1272 # 8 layers, attention
- 6: 8.4592 # 9 layers, attention

best network: 5
- seems converged
- sbc checks passed
- only `omicron_alpha` cannot be inferred for values above 0.2
- does not detect different recruitment scenarios in summary statistics
- attention on time pints seem to improve things a lot

# Both variants and trained only on PedCov
## Loss of networks:
- 7: 9.1994  # 6 layers, attention
- 8: 9.317 # 7 layers, attention
- 9: 8.977  # 8 layers, attention 

best network: 9
- seems converged
- sbc checks passed (only if random is dropped), but biased for tarp?
`omicron_alpha` still shows a lot of variance
- does not detect different recruitment scenarios in summary statistics
