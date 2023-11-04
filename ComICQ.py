import time
from intentAndActionExtraction import getIntent, getActions
from partsOfASentence import getObject, getSubject
from gloveSimilarity import calculate_similarity, load_word_vectors_from_file
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

reference_caption = "Three stuffed bears hugging and sitting on a blue pillow"
generated_caption = "A brown teddy bear"


def remove_stopwords(sentence):
    stop_words = set(stopwords.words("english"))
    words = sentence.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)


def jaccard_similarity(sentence1, sentence2):
    # Tokenize the sentences into sets of words
    words1 = set(sentence1.lower().split())
    words2 = set(sentence2.lower().split())

    # Calculate the Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1) + len(words2) - intersection
    similarity = intersection / union

    return similarity

def matchActions(referenceArray, genArray):
    referenceArray = [word.lower() for word in referenceArray]
    genArray = [word.lower() for word in genArray]
    matching_count = 0
    for word in genArray:
        if word in referenceArray:
            matching_count += 1
    return matching_count/len(referenceArray)


def novel_metric(reference_caption, generated_caption):
    reference_caption = remove_stopwords(reference_caption)
    generated_caption = remove_stopwords(generated_caption)

    refIntent = getIntent(reference_caption)
    genIntent = getIntent(generated_caption)

    refSubject = getSubject(refIntent)
    genSubject = getSubject(genIntent)

    print("Reference Subject:", refSubject)
    print("Gen Subject:", genSubject)

    refObject = getObject(refIntent)
    genObject = getObject(genIntent)

    print("Reference Object:", refObject)
    print("Gen Object:", genObject)

    word_vectors = load_word_vectors_from_file('word_vectors.pkl')
    semanticSubjectiveScore = calculate_similarity(refSubject, genSubject, word_vectors)
    semanticObjectiveScore = calculate_similarity(refObject, genObject, word_vectors)

    syntacticSubjectiveScore = jaccard_similarity(refSubject, genSubject)
    syntacticObjectiveScore = jaccard_similarity(refObject, genObject)

    subjectiveScore = 0.7 * semanticSubjectiveScore + 0.3 * syntacticSubjectiveScore
    objectiveScore = 0.7 * semanticObjectiveScore + 0.3 * syntacticObjectiveScore

    if genObject == "None" or genObject == "none" or refObject == "None" or refObject == "none":
        objectiveScore = 0
    if genSubject == "None" or genSubject == "none" or refSubject == "None" or refSubject == "none":
        subjectiveScore = 0


    time.sleep(60)

    refActions = getActions(reference_caption)
    genActions = getActions(generated_caption)

    actionScore = matchActions(refActions, genActions)

    print("Combined Subjective Score:", subjectiveScore)
    print("Combined Objective Score:", objectiveScore)
    print("Action Score:", actionScore)

    if len(refActions) == 0:
        if refObject == "None" or refObject == "none":
            joshi_score = subjectiveScore
        else:
            joshi_score = (objectiveScore+subjectiveScore)/2
    else:
        if refObject == "None" or refObject == "none":
            joshi_score = (objectiveScore + actionScore)/2
        else:
            joshi_score = (subjectiveScore + objectiveScore + actionScore) / 3

    return joshi_score


print(novel_metric(reference_caption, generated_caption))
