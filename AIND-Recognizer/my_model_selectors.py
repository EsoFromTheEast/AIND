import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_model = None
        lowest_BIC = float("inf")
        for state_num in range (self.min_n_components, self.max_n_components + 1):
            try :
                # Train Model
                model = self.base_model(state_num)

                # Calculate free parameters
                emission = model.means_.size + model.covars_.shape[0]*model.covars_.shape[1]
                transition = model.transmat_.size - model.transmat_.shape[0]
                initial = model.startprob_.size - 1
                freeParameters = initial + transition + emission

                # Calculate BIC Score
                logL = model.score(self.X,self.lengths)
                BIC = -2 * logL + freeParameters * math.log(model.n_features)

                # Store the lowest model
                if lowest_BIC > BIC :
                    lowest_BIC = BIC
                    best_model = model
            except:
                continue
        # Return minimum state number if model not found
        if best_model is None :
                    best_model = self.base_model(self.min_n_components)
        return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_model = None
        biggest_DIC = float("-inf")
        for state_num in range (self.min_n_components, self.max_n_components + 1):
            try :
                # Train Model
                model = self.base_model(state_num)
                logX = model.score(self.X,self.lengths)

                # Initialize logSum and Words count
                M = 0
                logSum = 0

                # For each word other than current word
                for word,XLengths in self.hwords.items() :
                    if word != self.this_word :
                        try :
                            # Calculate the Sum of Errors
                            logSum = logSum + model.score(XLengths[0],XLengths[1])
                            M += 1
                        except :
                            continue

                # Calculate DIC score
                M = max(1,M)
                DIC = logX - (logSum / M)

                # Store the biggest DIC score model
                if biggest_DIC < DIC :
                    biggest_DIC = DIC
                    best_model = model
            except:
                continue
        if best_model is None :
                    best_model = self.base_model(self.min_n_components)
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        split = min(len(self.sequences),10)
        folds = [[0,0]]
        if split > 1 :
            split_method = KFold(split)
            split_method.split(self.sequences)
            folds = split_method.split(self.sequences)
        # Intialize
        best_state = self.min_n_components
        sequences = np.array(self.sequences)
        logSums = np.zeros(self.max_n_components - self.min_n_components + 1)
        ModelNum = np.zeros(self.max_n_components - self.min_n_components + 1)
        x=lengths=x_test=lengths_test = list()
        for cv_train_idx, cv_test_idx in folds :
            try :
                x,lengths = combine_sequences(sequences[cv_train_idx])
                x_test,lengths_test = combine_sequences(sequences[cv_test_idx])
            except:
                x,lengths = self.X,self.lengths
                x_test,lengths_test = self.X,self.lengths
            for state_num in range (self.min_n_components, self.max_n_components + 1):
                try :
                    model = GaussianHMM(n_components=state_num, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(x, lengths)
                    logSums[state_num - self.min_n_components] += model.score(x_test,lengths_test)
                    ModelNum[state_num - self.min_n_components] += 1
                except:
                    continue

        # Handle states with 0 models
        logSums[ModelNum == 0]=float("-inf")
        ModelNum[ModelNum == 0] = 1
        averageLog = logSums / ModelNum

        return self.base_model(np.argmax(averageLog) + self.min_n_components)
