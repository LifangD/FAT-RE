









"""
Define constants.
"""
EMB_INIT_RANGE = 1.0

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1


VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]
MAX_LEN = 100
# hard-coded mappings from fields to ids
NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'DATE': 2, 'LOCATION': 3, 'SET': 4, 'MISC': 5, 'TIME': 6, 'CAUSE_OF_DEATH': 7, 'NATIONALITY': 8, 'CITY': 9, 'ORDINAL': 10, 'STATE_OR_PROVINCE': 11, 'TITLE': 12, 'DURATION': 13, 'CRIMINAL_CHARGE': 14, 'ORGANIZATION': 15, 'RELIGION': 16, 'NUMBER': 17, 'URL': 18, 'PERCENT': 19, 'COUNTRY': 20, 'IDEOLOGY': 21, 'O': 22, 'MONEY': 23, 'PERSON': 24}
OBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'DATE': 2, 'LOCATION': 3, 'SET': 4, 'MISC': 5, 'TIME': 6, 'CAUSE_OF_DEATH': 7, 'NATIONALITY': 8, 'CITY': 9, 'ORDINAL': 10, 'STATE_OR_PROVINCE': 11, 'TITLE': 12, 'DURATION': 13, 'CRIMINAL_CHARGE': 14, 'ORGANIZATION': 15, 'RELIGION': 16, 'NUMBER': 17, 'URL': 18, 'PERCENT': 19, 'COUNTRY': 20, 'IDEOLOGY': 21, 'O': 22, 'MONEY': 23, 'PERSON': 24}
SUBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'DATE': 2, 'LOCATION': 3, 'SET': 4, 'MISC': 5, 'TIME': 6, 'CAUSE_OF_DEATH': 7, 'NATIONALITY': 8, 'CITY': 9, 'ORDINAL': 10, 'STATE_OR_PROVINCE': 11, 'TITLE': 12, 'DURATION': 13, 'CRIMINAL_CHARGE': 14, 'ORGANIZATION': 15, 'RELIGION': 16, 'NUMBER': 17, 'URL': 18, 'PERCENT': 19, 'COUNTRY': 20, 'IDEOLOGY': 21, 'O': 22, 'MONEY': 23, 'PERSON': 24}


POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46}

DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'punct': 2, 'compound': 3, 'case': 4, 'nmod': 5, 'det': 6, 'nsubj': 7, 'amod': 8, 'conj': 9, 'dobj': 10, 'ROOT': 11, 'cc': 12, 'nmod:poss': 13, 'mark': 14, 'advmod': 15, 'appos': 16, 'nummod': 17, 'dep': 18, 'ccomp': 19, 'aux': 20, 'advcl': 21, 'acl:relcl': 22, 'xcomp': 23, 'cop': 24, 'acl': 25, 'auxpass': 26, 'nsubjpass': 27, 'nmod:tmod': 28, 'neg': 29, 'compound:prt': 30, 'mwe': 31, 'parataxis': 32, 'root': 33, 'nmod:npmod': 34, 'expl': 35, 'csubj': 36, 'cc:preconj': 37, 'iobj': 38, 'det:predet': 39, 'discourse': 40, 'csubjpass': 41}






INFINITY_NUMBER = 1e12



LABEL_TO_ID = {'Other': 0,'ORG-AFF':1, 'GEN-AFF':2, 'PART-WHOLE':3, 'PHYS':4, 'ART':5, 'PER-SOC':6}