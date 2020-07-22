import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from formulas import cosine_similarity, euclidean, get_country, get_accuracy, compute_pca

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

# check the accuracy of the model (Expected Output: ≈ 0.92)
# accuracy = get_accuracy(word_embeddings, data)
# print(f"Accuracy is {accuracy:.2f}")

# PCA check
np.random.seed(1)
X = np.random.rand(3, 10)
X_reduced = compute_pca(X, n_components=2)
print("Your original matrix was " + str(X.shape) + " and it became:")
print(X_reduced)

words = ['oil', 'gas', 'happy', 'sad', 'city', 'town',
         'village', 'country', 'continent', 'petroleum', 'joyful']

# given a list of words and the embeddings, it returns a matrix with all the embeddings
X = get_vectors(word_embeddings, words)

print('You have 11 words each of 300 dimensions thus X.shape is:', X.shape)

# Plotting PCA result
result = compute_pca(X, 2)
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0] - 0.05, result[i, 1] + 0.1))

plt.show()