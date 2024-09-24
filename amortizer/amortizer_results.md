
# Both variants and both selection methods
## Loss of networks:
- 0: 2.8628509044647217  # 6 layers
- 1: 1.9268755912780762 # 6 layers, attention (on time points)
- 2: 3.2145204544067383 # 7 layers
- 3: 1.8331129550933838 # 7 layers, attention
- 4: 2.5772318840026855 # 8 layers
- 5: 1.8832192420959473 # 8 layers, attention

best network: 3
- seems converged
- sbc checks passed
- does not detect different recruitment scenarios in summary statistics 
- attention on time pints seem to improve things quite a lot

# Both variants and trained only on PedCov
## Loss of networks:
- 6: 2.1603922843933105  # 6 layers, attention
- 7: 1.6322660446166992  # 7 layers, attention
- 8: 1.754188060760498  # 8 layers, attention 

best network: 7
- seems converged
- sbc checks passed
- does detect different recruitment scenarios in summary statistics for omicron
