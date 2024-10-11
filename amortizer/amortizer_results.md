
# Both variants and both selection methods
## Loss of networks:
- 0: 3.9909474849700928  # 6 layers
- 1: 3.3053367137908936  # 6 layers, attention (on time points)
- 2: 3.926607608795166 # 7 layers
- 3: 3.5322647094726562 # 7 layers, attention
- 4: 4.481245040893555 # 8 layers
- 5: 3.475659132003784 # 8 layers, attention

best network: 1
- seems converged
- sbc checks passed
- does detect different recruitment scenarios in summary statistics, why now?
- attention on time pints seem to improve things

# Both variants and trained only on PedCov
## Loss of networks:
- 7: 2.9431824684143066  # 6 layers, attention
- 8: 3.5736966133117676  # 7 layers, attention
- 9: 3.171257734298706  # 8 layers, attention 

best network: 7
- seems converged
- sbc checks passed (only if random is dropped)
- does detect different recruitment scenarios in summary statistics for omicron
