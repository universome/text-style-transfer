This repo contains some experiments on style transfer in text.

## TODO:
#### Current:
- temperature in inference
- cyclegan:
    - HPO for loss calibration
    - incorporate metrics
        - cosine distance between word2vecs
        - classifier confidence
        - BLEU with original sentences
        - prepare supervised corpus for validation
- mernn:
    - decoding with ACT
    - penalize early outputs more
- superconvergence

#### Future:
- lm discriminators:
    - better validation
    - replace transformer AE with RNN AE
- write clean summaries for each experiment
- HPO in firelab

#### Char word model:
- Add normal words to dataset, because it do not know usual words + overfits on gramma mistakes
- `__DROP__` token is being read character-level (it should be a single token)
- beam search
- Train model on 3 middle chars (which should be something like a root). Context is not needed in this case.
