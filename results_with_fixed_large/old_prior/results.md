# Results

Simulations: 
- 128*1000 = 128.000
- 10 epochs until batches are used up
- maximum of 250 epochs -> informs cosine decay

Amortizers:
inference networks are quite large (8 layers, 4 coupling layers each)
- `amortizer-sampling-bias-0`
  - only GRU as summary
  - sbc plots do not show any problems
  - training loss seems to have converged, the best validation loss: 6.066
- `amortizer-sampling-bias-0-conv`
  - GRU with 1d-convolution as summary
  - sbc plots do not show any problems
  - training loss seems to have converged, the best validation loss: 4.618 (best)
- `amortizer-sampling-bias-0-attention`
  - attention over households as summary
  - sbc plots do not show any problems
  - training loss seems to have converged, the best validation loss: 6.048
 - `amortizer-sampling-bias-0-conv-attention`
  - GRU with 1d-convolution and attention over households as summary
  - sbc plots do not show any problems
  - training loss seems to have converged, the best validation loss: 4.747
 - `amortizer-sampling-bias-0-conv-attention-bid`
  - bidirectional GRU with 1d-convolution and attention over households as summary
  - sbc plots do not show any problems
  - training loss seems to have converged, the best validation loss: 4.673
 - `amortizer-sampling-bias-0-conv-multi-attention-bid`
  - bidirectional GRU with 1d-convolution and multi head attention over households as summary
  - sbc plots okay, but posteriors are very wide
  - training loss seems not to have converged, stopped after only 30 epochs, the best validation loss: 6.399
 - `amortizer-sampling-bias-0-conv-attention-bid-MMD`
  - bidirectional GRU with 1d-convolution and attention over households as summary, MMD loss
  - sbc plots do not show any problems
  - training loss seems to have converged, the best validation loss: 4.782
 - `amortizer-sampling-bias-0-conv-bid-MMD`
  - bidirectional GRU with 1d-convolution, MMD loss
  - sbc plots do not show any problems
  - training loss seems to have converged, the best validation loss: 5.0778

Observations:
- convolutions seems to be a good idea
- attention only does not seem to be a good idea, attention needs more information, 
   so bidirectional GRU seems to be a good idea here
- multi head attention does not seem to be a good idea, does not converge at all
- MMD loss makes results only slightly worse
- plotting map against map seems to be similar to plotting median against median
