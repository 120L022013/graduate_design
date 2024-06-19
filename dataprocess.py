import json
import pickle
from collections import Counter
import re

MAX_VOCAB_SIZE = 10000                      # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'                 # 未知字，padding符号


def build_vocab_from_json(json_file, max_vocab_size):
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract text from each JSON object
    texts = [entry['text'] for entry in data]


    # Tokenize and flatten the texts
    tokens = [token for text in texts if
              isinstance(text, str) for token in re.split(r'[;,"@#/ .?$&*^!(){}\-]+', text) ]

    # Count the occurrences of each token
    token_counter = Counter(tokens)

    # Select the most common tokens up to the specified vocab size
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(token_counter.most_common(max_vocab_size - 2))}
    vocab['<UNK>'] = 0  # Unknown token
    vocab['<PAD>'] = 1  # Padding token

    # Save the vocabulary to a pickle file
    with open('./model/vocab.pkl', 'wb') as vocab_file:
        pickle.dump(vocab, vocab_file)

# Example usage
build_vocab_from_json('./data/output.json', MAX_VOCAB_SIZE)