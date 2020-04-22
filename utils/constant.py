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
SUBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1,  'PERSON': 2,'ORGANIZATION': 3}
#
OBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'PERSON': 2, 'ORGANIZATION': 3, 'DATE': 4, 'NUMBER': 5, 'TITLE': 6, 'COUNTRY': 7, 'LOCATION': 8, 'CITY': 9, 'MISC': 10, 'STATE_OR_PROVINCE': 11, 'DURATION': 12, 'NATIONALITY': 13, 'CAUSE_OF_DEATH': 14, 'CRIMINAL_CHARGE': 15, 'RELIGION': 16, 'URL': 17, 'IDEOLOGY': 18}

NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'O': 2, 'PERSON': 3, 'ORGANIZATION': 4, 'LOCATION': 5, 'DATE': 6, 'NUMBER': 7, 'MISC': 8, 'DURATION': 9, 'MONEY': 10, 'PERCENT': 11, 'ORDINAL': 12, 'TIME': 13, 'SET': 14}

POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46}

DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'punct': 2, 'compound': 3, 'case': 4, 'nmod': 5, 'det': 6, 'nsubj': 7, 'amod': 8, 'conj': 9, 'dobj': 10, 'ROOT': 11, 'cc': 12, 'nmod:poss': 13, 'mark': 14, 'advmod': 15, 'appos': 16, 'nummod': 17, 'dep': 18, 'ccomp': 19, 'aux': 20, 'advcl': 21, 'acl:relcl': 22, 'xcomp': 23, 'cop': 24, 'acl': 25, 'auxpass': 26, 'nsubjpass': 27, 'nmod:tmod': 28, 'neg': 29, 'compound:prt': 30, 'mwe': 31, 'parataxis': 32, 'root': 33, 'nmod:npmod': 34, 'expl': 35, 'csubj': 36, 'cc:preconj': 37, 'iobj': 38, 'det:predet': 39, 'discourse': 40, 'csubjpass': 41}

NEGATIVE_LABEL = 'no_relation'

LABEL_TO_ID = {'no_relation': 0, 'per:title': 1, 'org:top_members/employees': 2, 'per:employee_of': 3, 'org:alternate_names': 4, 'org:country_of_headquarters': 5, 'per:countries_of_residence': 6, 'org:city_of_headquarters': 7, 'per:cities_of_residence': 8, 'per:age': 9, 'per:stateorprovinces_of_residence': 10, 'per:origin': 11, 'org:subsidiaries': 12, 'org:parents': 13, 'per:spouse': 14, 'org:stateorprovince_of_headquarters': 15, 'per:children': 16, 'per:other_family': 17, 'per:alternate_names': 18, 'org:members': 19, 'per:siblings': 20, 'per:schools_attended': 21, 'per:parents': 22, 'per:date_of_death': 23, 'org:member_of': 24, 'org:founded_by': 25, 'org:website': 26, 'per:cause_of_death': 27, 'org:political/religious_affiliation': 28, 'org:founded': 29, 'per:city_of_death': 30, 'org:shareholders': 31, 'org:number_of_employees/members': 32, 'per:date_of_birth': 33, 'per:city_of_birth': 34, 'per:charges': 35, 'per:stateorprovince_of_death': 36, 'per:religion': 37, 'per:stateorprovince_of_birth': 38, 'per:country_of_birth': 39, 'org:dissolved': 40, 'per:country_of_death': 41}
OP_LABEL_TO_ID = {'no_relation': 0, 'per:title': 0, 'org:top_members/employees': 3, 'per:employee_of': 2, 'org:alternate_names': 4, 'org:country_of_headquarters': 0, 'per:countries_of_residence': 0, 'org:city_of_headquarters': 0, 'per:cities_of_residence': 0, 'per:age': 0, 'per:stateorprovinces_of_residence': 0, 'per:origin': 0, 'org:subsidiaries': 13, 'org:parents': 12, 'per:spouse': 14, 'org:stateorprovince_of_headquarters': 0, 'per:children': 22, 'per:other_family': 17, 'per:alternate_names': 18, 'org:members': 24, 'per:siblings': 20, 'per:schools_attended': 0, 'per:parents': 16, 'per:date_of_death': 0, 'org:member_of': 19, 'org:founded_by': 29, 'org:website': 0, 'per:cause_of_death': 0, 'org:political/religious_affiliation': 0, 'org:founded': 25, 'per:city_of_death': 0, 'org:shareholders': 0, 'org:number_of_employees/members': 0, 'per:date_of_birth': 0, 'per:city_of_birth': 0, 'per:charges': 0, 'per:stateorprovince_of_death': 0, 'per:religion': 0, 'per:stateorprovince_of_birth': 0, 'per:country_of_birth': 0, 'org:dissolved': 0, 'per:country_of_death':0}

sem_LABEL_TO_ID = {"Other": 0, "Cause-Effect(e1,e2)": 1, "Cause-Effect(e2,e1)": 2, "Component-Whole(e1,e2)": 3, "Component-Whole(e2,e1)": 4, "Content-Container(e1,e2)": 5, "Content-Container(e2,e1)": 6, "Entity-Destination(e1,e2)": 7, "Entity-Destination(e2,e1)": 8, "Entity-Origin(e1,e2)": 9, "Entity-Origin(e2,e1)": 10, "Instrument-Agency(e1,e2)": 11, "Instrument-Agency(e2,e1)": 12, "Member-Collection(e1,e2)": 13, "Member-Collection(e2,e1)": 14, "Message-Topic(e1,e2)": 15, "Message-Topic(e2,e1)": 16, "Product-Producer(e1,e2)": 17, "Product-Producer(e2,e1)": 18}
op_sem_LABEL_TO_ID = {"Other": 0, "Cause-Effect(e1,e2)": 2, "Cause-Effect(e2,e1)": 1, "Component-Whole(e1,e2)": 4, "Component-Whole(e2,e1)": 3, "Content-Container(e1,e2)": 6, "Content-Container(e2,e1)": 5, "Entity-Destination(e1,e2)": 8, "Entity-Destination(e2,e1)": 7, "Entity-Origin(e1,e2)": 10, "Entity-Origin(e2,e1)": 9, "Instrument-Agency(e1,e2)": 12, "Instrument-Agency(e2,e1)": 11, "Member-Collection(e1,e2)": 14, "Member-Collection(e2,e1)": 13, "Message-Topic(e1,e2)": 16, "Message-Topic(e2,e1)": 15, "Product-Producer(e1,e2)": 18, "Product-Producer(e2,e1)": 17}


INFINITY_NUMBER = 1e12

stopwords = ['!',',','.','?','-s','-ly','</s>','s','the','a','this','that']