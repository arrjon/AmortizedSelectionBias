
# Both variants and both selection methods
## Loss of networks:
- 0: 4.2119  # 6 layers
- 1: 3.5091  # 6 layers, attention (on time points)
- 2: 4.3851 # 7 layers
- 3: 3.3326 # 7 layers, attention
- 4: 4.2662 # 8 layers
- 5: 3.3932 # 8 layers, attention
- 6: 3.3452 # 9 layers, attention

best network: 3
- seems converged
- sbc checks passed (for all households)
- attention on time pints seem to improve things

# Both variants and trained only on PedCov
## Loss of networks:
- 7: 3.4812  # 6 layers, attention
- 8: 3.3088  # 7 layers, attention
- 9: 3.2647  # 8 layers, attention 

best network: 9
- seems converged
- sbc checks passed (only if random is dropped)
