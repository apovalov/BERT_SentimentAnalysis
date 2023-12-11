# Sentiment Analysis


![Alt text](/img/image.png)

# Satisfaction or Dissatisfaction

Customer satisfaction or dissatisfaction can help determine how high a restaurant should be in the ratings—it can be a make-or-break factor. If a restaurant from which we deliver has too many negative reviews, we must take measures to improve food quality or disconnect it from our service.

The task at hand is known as **Sentiment Analysis**, a text classification based on its sentiment:

- Positive (review)
- Negative (review)
- Neutral (review)

# Why are Transformers Called Transformers?

To answer this question, let's consider BERT, which is an encoder. Particularly, BERT is an excellent example of the transformer architecture because it outputs data in the same format that it takes as input, namely, embeddings (multidimensional representations of words or tokens; here, the term "word" and "token" are used interchangeably).

Embedding is, literally, a numerical vector of a word in multidimensional space, characterizing a word or token. If the input of the encoder also has an embedding size of 10 with a dimensionality of 128, then the output of the encoder will have the same 10 embeddings of dimensionality 128. However, after transforming the model, each embedding "attracts to itself" the meanings of the context (the words around it).

It turns out, we transform the abstract (isolated) meaning into the meaning of words "in context".

In other words, context-independent embeddings are transformed into context-conditioned ones.

![Alt text](/img/image-1.png)

**Embedding** is, literally, a numerical vector of a word in multidimensional space, characterizing a word or token. If the input to an encoder is also 10 embeddings of size 128, then the encoder's output will be 10 embeddings of dimensionality 128 as well. However, after the model transforms, each embedding "attracts to itself" the meanings of the context (words around it).

Thus, we transform the abstract (isolated) meaning of words into the meaning of words "in context".

In other words, context-independent embeddings are transformed into context-conditioned embeddings.


![Alt text](/img/image-2.png)

**Example:**
"Left: The dog did not cross the street because it was too tired. Right: The dog did not cross the street because it was too wide." The word "it" changes its meaning depending on the context, and the Self-Attention mechanism in transformers easily captures this connection.

A key feature of large transformers is that they are pretrained on one or several common tasks on a large corpus of texts, for instance, predicting the next word (Next Word Prediction), or MLM (Masked Language Modelling). We process them on "general understanding" of language, context, meaning. After this, we can use these models for our specialized tasks, sometimes without any additional training on our data (a stage called fine-tuning). This transfer of knowledge is known as Transfer Learning.

In particular, BERT is pretrained on two tasks simultaneously:

1. Masked Language Modelling (MLM)
2. Next Sentence Prediction (NSP)


# Masked Language Modelling

**Masked Language Modelling (MLM)**: We feed a sentence to the model, replacing some words with a "[MASK]" token, and ask it to predict the original text. To solve this task, BERT must understand the meaning of all other words around. During training, we adjust the model's parameters so that the embedding of the "[MASK]" token at the output is as close as possible to the original word's embedding.

Because we "mask" random tokens (which can be at any place in the text) and require their recovery based on what came before and what follows, BERT is designed for context-conditioned interpretation of individual words, utilizing the entire context, not just the preceding one. Therefore, it includes the word **Bidirectional** in its name (BERT – Bidirectional Encoder Representations from Transformers).

# Next Sentence Prediction

**Next Sentence Prediction (NSP)**: the task of predicting the next sentence. For this, two additional tokens are introduced, [CLS] and [SEP] (you may have noticed them in the images above). BERT takes two sentences as input, separated by [SEP], and is asked to classify whether the second sentence follows the first (IsNext) or not (NotNext).


# Fine-Tuning Pre-Trained Models Has Many Advantages:

- It is significantly faster in terms of time (a matter of hours vs. hundreds of hours).
- Requires substantially less labeled data (tailored for our specific task).
- As a consequence, often fine-tuning a pre-trained model yields better quality.

## Fine-tuning

Fine-tuning (for text classification) can be performed in various ways, for example:

1. You can take an already pre-trained [CLS] token and train a logistic regression on top of it.
2. You can freeze the first layers of the model and train parameters only for the last ones.
3. You can train all parameters, initializing the weights from the pre-training phase.

**Tokenizer in Language Models** - A tokenizer is a module that breaks down text into individual words (or tokens), which are then fed into a model as vectors. A tokenizer can split text by characters, syllables, parts of words, words, or entire phrases, depending on their occurrence in the training set and the task the model is designed to solve.

Inside the tokenizer, without our intervention:

1. Splits text into words and subwords (into tokens).
2. Adds special tokens: [CLS] and [SEP].
3. Converts words into indices (inside the model, a mapping index -> embedding is stored).

As a result, we obtain an index of words, natural for the model.

# Sentence Embedding
The code using your DataLoader will look something like the following:

![Alt text](/img/image-3.png)

```
from transformers import DistilBertModel, DistilBertTokenizer
from utils import DataLoader, attention_mask

MODEL_NAME = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
bert = DistilBertModel.from_pretrained(MODEL_NAME)

loader = DataLoader('./data/reviews.csv', tokenizer, max_limit=128, padding='batch')

for tokens, labels in loader:
    # Attention mask
    mask = attention_mask(tokens)

    # Calculate embeddings
    tokens = torch.tensor(tokens)
    mask = torch.tensor(mask)

    with torch.no_grad():
        last_hidden_states = bert(tokens, attention_mask=mask)

    # Embeddings for [CLS]-tokens
    features = last_hidden_states[0][:,0,:].tolist()
```

Here we iterate through the batches of our dataset, get the tokens (already padded) and class labels.
Next, we set the attention mask and apply BERT to our batches.

![Alt text](/img/image-4.png)

Finally, we take the last hidden state's [CLS] token, which represents sentence embedding for the entire string.
This is how we obtain embeddings of reviews!
Wrap the processing of one batch (as shown above) into the `review_embedding` function:

```python
def review_embedding(tokens: List[List[int]], model) -> List[List[float]]:
    """Return embedding for batch of tokenized texts"""
    ...
```

P.S. The option `torch.no_grad()` disables gradient calculation during the data passing through the model. This is necessary when we are performing validation or inference (which is also calculation of embeddings without training).


![Alt text](/img/image-5.png)

# Logistic Regression

We have just extracted our numerical representations of reviews (embeddings) using a pre-trained transformer. We now proceed to the final stage - training a model on top of these embeddings. For our task (multi-class classification of 3 classes), we will use the standard logistic regression from scikit-learn.


1. Tokenization
2. Batch-processing
3. Padding + Attention mask
4. Tokens -> Embeddings // multitask
5. CLS -> LogisticRegression