# Results

Simulations: 
- 128*1000 = 128.000
- 1 epoch until batches are used up
- maximum of 100 epochs -> informs cosine decay

Amortizers:
inference networks are quite large (8 layers, 4 coupling layers each)
- `amortizer-sampling-bias-0-lstm`
  - only LSTM as summary
  - recovery bad for all parameters
  - sbc plots okay
  - training loss seems to have converged
- `amortizer-sampling-bias-0-lstm-conv`
  - LSTM with 1d-convolution as summary
  - recovery seems a little better than before (at least for some parameters), but still bad
  - sbc plots okay
  - training loss does not seem to have converged yet (why did it stop?)
 - `amortizer-sampling-bias-0-lstm-conv-attention`
  - LSTM with 1d-convolution and attention over households as summary
  - recovery seems a little better than before (at least for some parameters), but still bad
  - sbc plots okay
  - training loss seems to have converged
 - `amortizer-sampling-bias-0-lstm-conv-attention-sum`
  - LSTM with 1d-convolution and attention over households as summary, sum pooling instead of max pooling
  - recovery the best so far?! (at least for some parameters), but still bad
  - sbc plots okay
  - training loss seems to have converged (but was way higher in the beginning)
 - `amortizer-sampling-bias-0-lstm-conv-attention-no-decay`
  - LSTM with 1d-convolution and attention over households as summary, no cosine decay of the learning rate
  - recovery bad for all parameters
  - sbc plots okay
  - training loss seems to have converged

Next steps:
- generate more simulations (10 epochs minimum)
- attention seems to improve things, also sum pooling instead of max pooling
