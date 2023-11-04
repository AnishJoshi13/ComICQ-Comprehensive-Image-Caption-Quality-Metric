import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import string
import pickle

# Load GloVe word vectors
def load_glove_vectors(glove_file):
    word_vectors = {}
    print("Loading word vectors...")
    with open(glove_file, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.strip().split(' ')
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            word_vectors[word] = vector
    return word_vectors

# Function to save word vectors to a file
def save_word_vectors_to_file(word_vectors, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(word_vectors, file)

# Function to load word vectors from a file
def load_word_vectors_from_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            word_vectors = pickle.load(file)
        return word_vectors
    except (FileNotFoundError, EOFError):
        return {}

# Preprocess a sentence: convert to lowercase, remove punctuation, and stopwords
def preprocess_sentence(sentence):
    stop_words = set(stopwords.words('english'))
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    words = sentence.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Calculate sentence vector as the mean of word vectors
def sentence_vector(sentence, word_vectors):
    words = sentence.split()
    vectors = [word_vectors[word] for word in words if word in word_vectors]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(len(vectors[0]))

# Calculate cosine similarity between two sentences
def calculate_similarity(sentence1, sentence2, word_vectors):
    sentence1 = preprocess_sentence(sentence1)
    sentence2 = preprocess_sentence(sentence2)
    vector1 = sentence_vector(sentence1, word_vectors)
    vector2 = sentence_vector(sentence2, word_vectors)
    if np.any(vector1) and np.any(vector2):
        similarity = cosine_similarity([vector1], [vector2])
        return similarity[0][0]
    else:
        return 0.0  # Return 0 if either sentence is empty

if __name__ == '__main__':
    glove_file = r"C:\Users\Anish Joshi\Downloads\glove.840B.300d\glove.840B.300d.txt"  # Path to your GloVe file
    word_vectors = load_word_vectors_from_file('word_vectors.pkl')

    if not word_vectors:
        # If word_vectors were not found in the file, load them from the original GloVe file and save them
        word_vectors = load_glove_vectors(glove_file)
        save_word_vectors_to_file(word_vectors, 'word_vectors.pkl')

    sentence1 = "Three stuffed bears hugging and sitting on a blue pillow"
    sentence2 = "A brown teddy bear"

    similarity = calculate_similarity(sentence1, sentence2, word_vectors)
    print("Similarity:", similarity)
