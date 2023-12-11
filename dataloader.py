from dataclasses import dataclass
from typing import Generator
from typing import List
from typing import Tuple

import torch
from transformers import PreTrainedTokenizer


@dataclass
class DataLoader:
    path: str
    tokenizer: PreTrainedTokenizer
    batch_size: int = 512
    max_length: int = 128
    padding: str = None

    def __iter__(self) -> Generator[List[List[int]], None, None]:
        """Iterate over batches"""
        for i in range(len(self)):
            yield self.batch_tokenized(i)

    def __len__(self):
        """Number of batches"""
        # Number of rows
        with open(self.path) as f:
            n_rows = sum(1 for _ in f)
        n_rows -= 1  # Skip header

        # Round up
        return n_rows // self.batch_size + bool(n_rows % self.batch_size)

    def tokenize(self, batch: List[str]) -> List[List[int]]:
        """Tokenize list of texts"""
        output = []
        for text in batch:
            tokenized = self.tokenizer.encode(
                text,
                max_length=self.max_length,
                add_special_tokens=True,
                truncation=True,
            )
            output.append(tokenized)

        if self.padding == "max_length":
            max_len = self.max_length
            for i, x in enumerate(output):
                output[i] = x + [0] * (max_len - len(x))

        elif self.padding == "batch":
            max_len = max(len(x) for x in output)
            for i, x in enumerate(output):
                output[i] = x + [0] * (max_len - len(x))

        return output

    def batch_loaded(self, i: int) -> Tuple[List[str], List[int]]:
        """Return loaded i-th batch of data (text)"""
        index_start = i * self.batch_size
        index_end = (i + 1) * self.batch_size

        texts = []
        labels = []
        with open(self.path) as f:
            _ = next(f)  # Skip header
            for j, line in enumerate(f):
                if index_start <= j < index_end:
                    fields = line.split(",", 4)

                    sentiment = fields[3]
                    if sentiment == "positive":
                        label = 1
                    elif sentiment == "negative":
                        label = -1
                    else:
                        label = 0

                    texts.append(fields[4].strip())
                    labels.append(label)

                if j >= index_end:
                    break

        return texts, labels

    def batch_tokenized(self, i: int) -> Tuple[List[List[int]], List[int]]:
        """Return tokenized i-th batch of data"""
        texts, labels = self.batch_loaded(i)
        tokens = self.tokenize(texts)
        return tokens, labels


def attention_mask(batch: List[List[int]]) -> List[List[int]]:
    """Return attention mask for batch of tokenized texts"""
    mask = []
    for tokens in batch:
        mask.append([1 if x != 0 else 0 for x in tokens])
    return mask


def review_embedding(tokens: List[List[int]], model) -> List[List[float]]:
    """Return embedding for batch of tokenized texts"""
    # Attention mask
    mask = attention_mask(tokens)

    # Calculate embeddings
    tokens = torch.tensor(tokens)
    mask = torch.tensor(mask)

    with torch.no_grad():
        embeddings = model(tokens, attention_mask=mask)

    # Extract CLS embeddings
    features = embeddings[0][:, 0, :].tolist()

    return features
