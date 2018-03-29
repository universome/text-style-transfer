Initial transformer implementation is taken from [jadore801120](https://github.com/jadore801120/attention-is-all-you-need-pytorch)


TODO:
* Validate that fast inference produces the same probabilities as the slow one. We have a problem here cause we slow inference produces different probabilities (in `eval` mode) and we should increase our `atol` in `np.allclose` during checks. But divergence between fast and slow inference was superbig :|