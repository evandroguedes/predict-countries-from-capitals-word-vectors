# Run this cell to import packages.
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from formulas import cosine_similarity, euclidean, get_country

from utils import get_vectors

data = pd.read_csv('capitals.txt', delimiter=' ')
data.columns = ['city1', 'country1', 'city2', 'country2']

# print first five elements in the DataFrame
print(data.head(5))

word_embeddings = pickle.load(open('word_embeddings_subset.p', 'rb'))
print('word embeddings length:', len(word_embeddings))  # there should be 243 words that will be used in this assignment

# Each of the word embedding is a 300-dimensional vector.
print('dimension: {}'.format(word_embeddings['Spain'].shape[0]))

king = word_embeddings['king']
queen = word_embeddings['queen']

# expected output ≈ 0.6510956
print('cosine similarity:', cosine_similarity(king, queen))

#Expected Output: 2.4796925
print('euclidean distance:', euclidean(king, queen))

# Testing your function, note to make it more robust you can return the 5 most similar words.
# Expected Output: ('Egypt', 0.7626821)
print(get_country('Athens', 'Greece', 'Cairo', word_embeddings))