import numpy as np
import matplotlib.pyplot as plt

from src.trainers import UMTTrainer
from src.utils.data_utils import pad_to_longest, token_ids_to_sents
from src.vocab import constants
from src.utils.bleu import compute_bleu_for_sents


class StyleTransferTrainer(UMTTrainer):
    def __init__(self, *args, **kwargs):
        super(StyleTransferTrainer, self).__init__(*args, **kwargs)

        self.log_file = args[-1].get('log_file')
        self.val_scores = {
            'bleu_src_to_trg_to_src': [],
            'bleu_trg_to_src_to_trg': []
        }

    def validate_bleu(self, val_data, return_results=False, beam_size=1):
        self.transformer.eval()
        sources = []
        targets = []
        translations_src_to_trg = []
        translations_trg_to_src = []
        translations_src_to_trg_to_src = []
        translations_trg_to_src_to_trg = []

        for src, trg in val_data:
            src_to_trg = self.transformer.translate_batch(
                src, 'trg', max_len=self.max_seq_len, beam_size=beam_size)
            trg_to_src = self.transformer.translate_batch(
                trg, 'src', max_len=self.max_seq_len, beam_size=beam_size)

            src_to_trg_var = pad_to_longest([[constants.BOS] + s for s in src_to_trg], volatile=True)
            trg_to_src_var = pad_to_longest([[constants.BOS] + s for s in trg_to_src], volatile=True)

            src_to_trg_to_src = self.transformer.translate_batch(
                src_to_trg_var, 'src', max_len=self.max_seq_len, beam_size=beam_size)
            trg_to_src_to_trg = self.transformer.translate_batch(
                trg_to_src_var, 'trg', max_len=self.max_seq_len, beam_size=beam_size)

            sources += token_ids_to_sents(src, self.vocab_src)
            targets += token_ids_to_sents(trg, self.vocab_trg)

            translations_src_to_trg += token_ids_to_sents(src_to_trg, self.vocab_trg)
            translations_trg_to_src += token_ids_to_sents(trg_to_src, self.vocab_src)

            translations_src_to_trg_to_src += token_ids_to_sents(src_to_trg_to_src, self.vocab_src)
            translations_trg_to_src_to_trg += token_ids_to_sents(trg_to_src_to_trg, self.vocab_trg)

        bleu_src_to_trg_to_src = compute_bleu_for_sents(translations_src_to_trg_to_src, sources)
        bleu_trg_to_src_to_trg = compute_bleu_for_sents(translations_trg_to_src_to_trg, targets)

        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write('Epochs done: {}. Iters done: {}\n'.format(self.num_epochs_done, self.num_iters_done))
                f.write('[src->trg->src] BLEU: {}. [trg->src->trg] BLEU: {}.\n'.format(bleu_src_to_trg_to_src, bleu_trg_to_src_to_trg))

                f.write('##### SRC->TRG translations #####\n')
                for i, src_to_trg in enumerate(translations_src_to_trg):
                    f.write('Source    : ' + sources[i] + '\n')
                    f.write('Translated: ' + src_to_trg + '\n')

                f.write('\n')
                f.write('##### TRG->SRC translations #####\n')

                for i, trg_to_src in enumerate(translations_trg_to_src):
                    f.write('Target:          ' + targets[i] + '\n')
                    f.write('Translated target: ' + trg_to_src + '\n')

                f.write('\n===============================================================================\n')

        if return_results:
            scores = (bleu_src_to_trg_to_src, bleu_trg_to_src_to_trg)
            translations = {
                'sources': sources,
                'targets': targets,
                'translations_src_to_trg': translations_src_to_trg,
                'translations_trg_to_src': translations_trg_to_src,
                'translations_src_to_trg_to_src': translations_src_to_trg_to_src,
                'translations_trg_to_src_to_trg': translations_trg_to_src_to_trg
            }

            return scores, translations
        else:
            self.val_scores['bleu_src_to_trg_to_src'].append(bleu_src_to_trg_to_src)
            self.val_scores['bleu_trg_to_src_to_trg'].append(bleu_trg_to_src_to_trg)

    def plot_validation_scores(self):
        if not self.val_scores['bleu_src_to_trg_to_src']: return

        plt.figure(figsize=[16,4])

        src, trg = 'bleu_src_to_trg_to_src', 'bleu_trg_to_src_to_trg'
        val_iters = np.arange(len(self.val_scores[src])) * (self.num_iters_done / len(self.val_scores[src]))

        plt.title('Val translation BLEU')
        plt.plot(val_iters, self.val_scores[src], label='BLEU score for [src->trg->src]')
        plt.plot(val_iters, self.val_scores[trg], label='BLEU score for [trg->src->trg]')
        plt.grid()
        plt.legend()

        plt.show()
