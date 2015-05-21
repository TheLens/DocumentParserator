# -*- coding: utf-8 -*-
import pycrfsuite
import os
import warnings
from collections import OrderedDict
import re
import nltk
import pdb
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import words


# http://stackoverflow.com/questions/2150205/can-somebody-explain-a-money-regex-that-just-checks-if-the-value-matches-some-pa
CURRENCY_REG = money = re.compile('|'.join([
  r'^\$?(\d*\.\d{1,2})$',  # e.g., $.50, .50, $1.50, $.5, .5
  r'^\$?(\d+)$',           # e.g., $500, $5, 500, 5
  r'^\$(\d+\.?)$',         # e.g., $5.
]))

#  _____________________
# |1. CONFIGURE LABELS! |
# |_____________________| 
#     (\__/) || 
#     (•ㅅ•) || 
#     / 　 づ

LABELS = ["amendment_amount", "amendment_amount_padding", "contract_amount", "contract_amount_padding", "other_amount", "other_amount_padding", "amount_alphabetic", "skip", "post_amended_amount", "post_amended_amount_padding"] # The labels should be a list of strings


def get_number_words():
    return [l.replace("\n", "").lower() for l in open('number_words.txt')]

NUMBER_WORDS = get_number_words()

#***************** OPTIONAL CONFIG ***************************************************
PARENT_LABEL  = 'amount_string'               # the XML tag for each labeled string
GROUP_LABEL   = 'Collection'                  # the XML tag for a group of strings
NULL_LABEL    = 'Null'                        # the null XML tag
MODEL_FILE    = 'learned_settings.crfsuite'   # filename for the crfsuite settings file
#************************************************************************************

try :
    TAGGER = pycrfsuite.Tagger()
    TAGGER.open(os.path.split(os.path.abspath(__file__))[0]+'/'+MODEL_FILE)
except IOError :
    TAGGER = None
    warnings.warn('You must train the model (parserator train [traindata] [modulename]) to create the %s file before you can use the parse and tag methods' %MODEL_FILE)

def parse(raw_string):
    if not TAGGER:
        raise IOError('\nMISSING MODEL FILE: %s\nYou must train the model before you can use the parse and tag methods\nTo train the model annd create the model file, run:\nparserator train [traindata] [modulename]' %MODEL_FILE)
    tokenizer = RegexpTokenizer('\$[\d]+(\.\d)? billion|\w+|\$[\d\,]+(.\d\d)?')
    tokens = tokenizer.tokenize(raw_string)
    if not tokens :
        return []

    features = tokens2features(tokens)

    tags = TAGGER.tag(features)
    return zip(tokens, tags)


def tag(raw_string) :
    tagged = OrderedDict()
    for token, label in parse(raw_string) :
        tagged.setdefault(label, []).append(token)

    for token in tagged :
        component = ' '.join(tagged[token])
        component = component.strip(' ,;')
        tagged[token] = component

    return tagged


#  _____________________
# |2. CONFIGURE TOKENS! |
# |_____________________| 
#     (\__/) || 
#     (•ㅅ•) || 
#     / 　 づ
def tokenize(raw_string, offsets = False):
    if not offsets: # standard run of tokenization
        tokenizer = RegexpTokenizer('\$[\d]+(\.\d)? billion|\w+|\$[\d\,]+(.\d\d)?')
        tokens = tokenizer.tokenize(raw_string)
        if not tokens :
            return []
        return tokens
    else: # tokenization by returning offsets
        output = []
        for i in re.finditer('\$[\d]+(\.\d)? billion|\w+|\$[\d\,]+(.\d\d)?',raw_string):
            output.append((i.start(), i.end()))
        return output

#  _______________________
# |3. CONFIGURE FEATURES! |
# |_______________________| 
#     (\__/) || 
#     (•ㅅ•) || 
#     / 　 づ
def tokens2features(tokens):
    # this should call tokenFeatures to get features for individual tokens,
    # as well as define any features that are dependent upon tokens before/after
    
    feature_sequence = [tokenFeatures(tokens[0])]
    previous_features = feature_sequence[-1].copy()

    for token in tokens[1:] :
        # set features for individual tokens (calling tokenFeatures)
        token_features = tokenFeatures(token)
        current_features = token_features.copy()

        # features for the features of adjacent tokens
        feature_sequence[-1]['next'] = current_features
        token_features['previous'] = previous_features        
        
        # DEFINE ANY OTHER FEATURES THAT ARE DEPENDENT UPON TOKENS BEFORE/AFTER
        # for example, a feature for whether a certain character has appeared previously in the token sequence
        
        feature_sequence.append(token_features)
        previous_features = current_features

    if len(feature_sequence) > 1 :
        # these are features for the tokens at the beginning and end of a string
        feature_sequence[0]['rawstring.start'] = True
        feature_sequence[-1]['rawstring.end'] = True
        feature_sequence[1]['previous']['rawstring.start'] = True
        feature_sequence[-2]['next']['rawstring.end'] = True

    else : 
        # a singleton feature, for if there is only one token in a string
        feature_sequence[0]['singleton'] = True

    return feature_sequence

def tokenFeatures(token) :
    # this defines a dict of features for an individual token

    features = {   # DEFINE FEATURES HERE. some examples:
                    'is_all_upper'  : is_all_upper(token),
                    'token' : token.lower(),
                    'has_dollar_sign' : has_dollar_sign(token),
                    'has_number_word' : has_number_word(token),
                    'can_convert_to_float' : can_convert_to_float(token),
                    'has_comma' : has_comma(token), 
                    'has_slash' : has_slash(token),
                    'is_english' : is_english(token),
                    'is_red_herring' : is_red_herring(token),
                    'num_alpha' : num_alpha(token),
                    'has_numeric_token' : has_numberic_token(token)
                }

    return features


def num_alpha(token):
    num_alphas = len([i for i in token if i.isdigit()])
    return num_alphas


def is_red_herring(token):
    herings = ["per", "insurance"]
    if token in herings:
        return True
    else:
        return False


def is_currency(token):
    return CURRENCY_REG.match(token)


def is_english(token):
    if token in words.words():
        return True
    else:
        return False


def has_slash(token):
    if "/" in token:
        return True
    else: 
        return False

def has_comma(token):
    if "," in token:
        return True
    else: 
        return False

def contains_billion(token):
    if "billion" in token:
        return True
    else:
        return False

def can_convert_to_float(token):
    token = token.replace(",", "").replace("$", "")
    try:
        float(token)
        return True
    except:
        return False

def has_dollar_sign(token):
    if "$" in token:
        return True
    else:
        return False

def has_number_word(token):
    if token.lower() in NUMBER_WORDS:
        return True
    else:
        return False

def has_numberic_token(token):
    return any(char.isdigit() for char in token)


# define any other methods for features. this is an example to get the casing of a token
def is_all_upper(token) :
    if token.isupper() :
        return True
    else :
        return False
