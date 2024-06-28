# Results

Simulations: 
- 128*1000 = 128.000
- 1 epoch until batches are used up
- maximum of 500 epochs (reduced for next training round to 200) -> informs cosine decay

Amortizers:
- `amortizer-sampling-bias-0`
  - small inference network, only GRU as summary, sbc plots okay, but recovery bad for all parameters
  - stopped after 15 epochs
- `amortizer-sampling-bias-0-lstm`
  - small inference network, only LSTM as summary, sbc plots okay, but recovery bad for all parameters
  - not really better than GRU
  - stopped after 23 epochs
- `amortizer-sampling-bias-0-lstm-conv`
  - LSTM with 1d-convolution as summary, sbc plots okay, but recovery bad for all parameters
  - recovery seems a little better than before
  - stopped after 107 epochs
- `amortizer-sampling-bias-0-sum`
  - sbc plots less okay
  - stopped after 41 epochs
- `amortizer-sampling-bias-1-lstm-conv-spec`
  - larger inference network, seems not to be able to learn
  - stopped after 37 epochs

Next steps:
- generate more simulations (10 epochs minimum)
- lstm with conv seems to be the best choice so far
- log transform parameters (in configurator)
- maybe try without early stopping