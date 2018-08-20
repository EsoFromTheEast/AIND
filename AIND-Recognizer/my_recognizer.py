import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    sequences = test_set.get_all_sequences()
    Xlengths = test_set.get_all_Xlengths()
    # For each word indexed in the DataFrame
    for word_id,word in enumerate(test_set.wordlist):
        probability = dict()
        X,lengths = test_set.get_item_Xlengths(word_id)
        # For each model
        for model_word,model in models.items():
            try :
                # Calculate the score and store it in probability List
                probability[model_word] = model.score(X,lengths)
            except :
                probability[model_word] = float("-inf")
        # Append the word's probability list to the probabilities result list
        probabilities.append(probability)
        # Append the word with maximum probability to the guess result list
        guesses.append(max(probability.keys(), key=(lambda key: probability[key])))
    # return probabilities, guesses
    return (probabilities,guesses)
    