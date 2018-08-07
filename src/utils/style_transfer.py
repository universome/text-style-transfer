from src.utils.data_utils import itos_many


def transfer_style(transfer_style_on_batch, dataloader, vocab):
    """
    Produces predictions for a given dataloader
    """
    domain_x_to_domain_y = []
    domain_y_to_domain_x = []
    domain_x_to_domain_x = []
    domain_y_to_domain_y = []
    gold_domain_x = []
    gold_domain_y = []

    for batch in dataloader:
        x2y, y2x, x2x, y2y = transfer_style_on_batch(batch)

        domain_x_to_domain_y.extend(x2y)
        domain_y_to_domain_x.extend(y2x)
        domain_x_to_domain_x.extend(x2x)
        domain_y_to_domain_y.extend(y2y)

        gold_domain_x.extend(batch.domain_x.detach().cpu().numpy().tolist())
        gold_domain_y.extend(batch.domain_y.detach().cpu().numpy().tolist())

    # Converting to sentences
    x2y_sents = itos_many(domain_x_to_domain_y, vocab)
    y2x_sents = itos_many(domain_y_to_domain_x, vocab)
    x2x_sents = itos_many(domain_x_to_domain_x, vocab)
    y2y_sents = itos_many(domain_y_to_domain_y, vocab)
    gx_sents = itos_many(gold_domain_x, vocab)
    gy_sents = itos_many(gold_domain_y, vocab)

    return x2y_sents, y2x_sents, x2x_sents, y2y_sents, gx_sents, gy_sents


def get_text_from_sents(x2y_s, y2x_s, x2x_s, y2y_s, gx_s, gy_s):
    # TODO: Move this somewhere from trainer file? Or create some nice template?
    return """
        Gold X: [{}]
        Gold Y: [{}]

        x2y: [{}]
        y2x: [{}]

        x2x: [{}]
        y2y: [{}]
    """.format(gx_s, gy_s, x2y_s, y2x_s, x2x_s, y2y_s)
