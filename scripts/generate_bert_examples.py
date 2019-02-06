from typing import Set

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-models/pytorch_model.tar.gz')
model.eval()
positive_words = set([w.lower() for w in open('data/yelp/words-positive.txt').read().splitlines()])


def predict(sentences:str) -> str:
    tokenized = [tokenizer.tokenize(s) for s in sentences]
    masks_idx = [[i for i,w in enumerate(s) if w.lower() in positive_words] for s in tokenized]

    for j, mask in enumerate(masks_idx):
        for i in mask:
            tokenized[j][i] = '[MASK]'

    tokenized = [s for i, s in enumerate(tokenized) if len(masks_idx[i]) > 0]
    masks_idx = [s for i, s in enumerate(masks_idx) if len(masks_idx[i]) > 0]

    tokens_idx = [tokenizer.convert_tokens_to_ids(s) for s in tokenized]
    segments_ids = [[0] * len(s) for s in tokens_idx]

    # Convert inputs to PyTorch tensors
    tokens_tensors = [torch.tensor([s]) for s in tokens_idx]
    segments_tensors = [torch.tensor([s]) for s in segments_ids]

    predictions = [model(tokens, segments) for tokens, segments in tqdm(zip(tokens_tensors, segments_tensors))]
    predicted_idx = [p[0][idx].argmax(dim=0).numpy().tolist() for p, idx in zip(predictions, masks_idx)]
    predicted_tokens = [tokenizer.convert_ids_to_tokens(p) for p in predicted_idx]

    for j, s in enumerate(tokenized):
        for i, pred in zip(masks_idx[j], predicted_tokens[j]):
            s[i] = pred

    original = [tokenizer.tokenize(s) for s in sentences]

    for orig, result in zip(original, tokenized):
        print('From:', ' '.join(orig))
        print('To  :', ' '.join(result))
        print()

sentences = open('data/yelp/yelp-positive-short.en').read().splitlines()
sentences = list(filter(len, sentences))
sentences = sentences[:10]
predict(sentences)
