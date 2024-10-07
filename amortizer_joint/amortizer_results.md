
# Both variants jointly and both selection methods
## Loss of networks:
- 0: 11.92  # 6 layers
- 1: 8.748  # 6 layers, attention (on time points)
- 2: 11.399 # 7 layers
- 3: 8.447 # 7 layers, attention
- 4: 11.936 # 8 layers
- 5: 8.089 # 8 layers, attention

best network: 5
- seems converged
- sbc checks passed
- only `omicron_alpha` cannot be inferred for values above 0.2
- does not detect different recruitment scenarios in summary statistics
- attention on time pints seem to improve things a lot

# Both variants and trained only on PedCov
## Loss of networks:
- 6: 8.4633  # 6 layers, attention
- 7: 8.6957  # 7 layers, attention
- 8: 8.7220  # 8 layers, attention 

best network: 6
- seems converged
- sbc checks passed (only if random is dropped)
- only `omicron_alpha` cannot be inferred for values above 0.2
- does not detect different recruitment scenarios in summary statistics
