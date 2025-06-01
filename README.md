# BBC_News_Archive_with_NLP
# Text Classification and Generation with TensorFlow

This repository contains implementations of text processing, classification, and generation using TensorFlow. The project demonstrates three main functionalities:
1. Text classification on BBC news articles
2. Sentiment analysis on tweets
3. Shakespearean sonnet generation using LSTM

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [BBC Text Classification](#bbc-text-classification)
  - [Tweet Sentiment Analysis](#tweet-sentiment-analysis)
  - [Sonnet Generation](#sonnet-generation)
- [Project Structure](#project-structure)
- [Results](#results)
- [Visualization](#visualization)
- [TensorFlow Embedding Projector](#tensorflow-embedding-projector)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Text Preprocessing**: Standardization, stopword removal, tokenization
- **Text Classification**: Categorize BBC news articles into topics
- **Sentiment Analysis**: Classify tweet sentiment (positive/negative)
- **Text Generation**: Generate Shakespeare-like sonnets using LSTM
- **Embedding Visualization**: Compatible with TensorFlow Embedding Projector

## Usage
BBC Text Classification
Classifies BBC news articles into categories (sport, business, politics, tech, entertainment).

python
from bbc_classification import parse_data_from_file, standardize_func, fit_vectorizer

# Load and preprocess data
sentences, labels = parse_data_from_file("./data/bbc-text.csv")
standard_sentences = [standardize_func(sentence) for sentence in sentences]

# Vectorize text
vectorizer = fit_vectorizer(standard_sentences)
padded_sequences = vectorizer(standard_sentences)

# Train model (see notebook for complete example)
Tweet Sentiment Analysis
Classifies tweets as positive or negative sentiment.

python
from sentiment_analysis import train_val_datasets, fit_vectorizer, create_model

# Load and prepare dataset
dataset = tf.data.Dataset.from_tensor_slices((sentences, labels))
train_dataset, validation_dataset = train_val_datasets(dataset)

# Vectorize and train
vectorizer = fit_vectorizer(train_dataset.map(lambda text, label: text))
model = create_model(vectorizer.vocabulary_size(), embeddings_matrix)
history = model.fit(train_dataset_vectorized, epochs=20)
Sonnet Generation
Generates Shakespeare-like sonnets using LSTM.

python
from sonnet_generation import n_gram_seqs, pad_seqs, features_and_labels_dataset, create_model

# Prepare dataset
input_sequences = n_gram_seqs(corpus, vectorizer)
padded_sequences = pad_seqs(input_sequences, max_sequence_len)
dataset = features_and_labels_dataset(padded_sequences, total_words)

# Train model
model = create_model(total_words, max_sequence_len)
history = model.fit(dataset, epochs=30)

# Generate text
seed_text = "Shall I compare thee"
generated_text = generate_text(seed_text, next_words=100)

```
Project Structure
text-processing-tensorflow/
├── data/                   # Dataset files
│   ├── bbc-text.csv        # BBC news articles dataset
│   ├── training_cleaned.csv # Cleaned Sentiment140 tweets dataset
│   ├── sonnets.txt         # Shakespeare’s sonnets
│   └── glove.6B.100d.txt   # Pre-trained GloVe embeddings
├── notebooks/              # Jupyter notebooks for experiments
│   ├── bbc_classification.ipynb
│   ├── sentiment_analysis.ipynb
│   └── sonnet_generation.ipynb
├── src/                    # Source code for core functionalities
│   ├── bbc_classification.py
│   ├── sentiment_analysis.py
│   └── sonnet_generation.py
├── requirements.txt        # Python dependencies
└── README.md               # Project overview
```

Results
BBC Classification
Achieved high accuracy in categorizing news articles

Effective text preprocessing with stopword removal

Sentiment Analysis
Validation accuracy: ~84%

Uses pre-trained GloVe embeddings

Sonnet Generation
Training accuracy: ~87%

Generates coherent Shakespeare-like text

Visualization
Training performance visualized using matplotlib:

Training Performance

TensorFlow Embedding Projector
The project supports visualization of word embeddings using TensorFlow's Embedding Projector:

Save embeddings:

python
import tensorflow as tf
from tensorflow.keras.layers import Embedding

# Assuming 'embedding_layer' is your embedding layer
weights = embedding_layer.get_weights()[0]
vocab = vectorizer.get_vocabulary()

out_v = open('vectors.tsv', 'w', encoding='utf-8')
out_m = open('metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
    vec = weights[index]
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
    out_m.write(word + "\n")
out_v.close()
out_m.close()
Visualize in Embedding Projector:

Upload vectors.tsv and metadata.tsv to https://projector.tensorflow.org/

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
MIT License

## Required packages:

TensorFlow 2.x

pandas

numpy

matplotlib


## Installation

1. Clone the repository:
```bash
git clone https://github.com/TanvirArefin/BBC_News_Archive_with_NLP.git
cd BBC-News-Archive-with-NLP
Install dependencies:

bash
pip install -r requirements.txt
