import gensim
import pprint
from gensim import corpora,models
from gensim.utils import simple_preprocess
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec


doc_list = [
   "Natural language processing (NLP) is a subfield of Artificial Intelligence (AI).", 
   "This technology works on the speech provided by the user breaks it down for proper understanding and processes it accordingly.", 
   "This is a very recent and effective approach due to which it has a really high demand in today’s market. "
   "Natural Language Processing (NLP) is a subfield of artificial intelligence that deals with the interaction between computers and humans in natural language."
   
]

tokens1 = [[item for item in line.split()] for line in doc_list]
g_dict1 = corpora.Dictionary(tokens1)

print("The dictionary has: " +str(len(g_dict1)) + " tokens\n")
print(g_dict1.token2id)

g_bow =[g_dict1.doc2bow(token, allow_update = True) for token in tokens1]
print("Bag of Words : ", g_bow)



print("\n--------------------------------------------------------------------------------------------------------\n")


g_dict = corpora.Dictionary([simple_preprocess(line) for line in doc_list])
g_bow = [g_dict.doc2bow(simple_preprocess(line)) for line in doc_list]

print("Dictionary : ")
for item in g_bow:
    print([[g_dict[id], freq] for id, freq in item])

g_tfidf = models.TfidfModel(g_bow, smartirs='ntc')

print("\nTF-IDF Vector:")
for item in g_tfidf[g_bow]:
    print([[g_dict[id], np.around(freq, decimals=2)] for id, freq in item])


print("\n--------------------------------------------------------------------------------------------------------\n")


#Output:-

'''
import gensim
import pprint
from gensim import corpora,models
from gensim.utils import simple_preprocess
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec


doc_list = [
   "Natural language processing (NLP) is a subfield of Artificial Intelligence (AI).", 
   "This technology works on the speech provided by the user breaks it down for proper understanding and processes it accordingly.", 
   "This is a very recent and effective approach due to which it has a really high demand in today’s market. "
   "Natural Language Processing (NLP) is a subfield of artificial intelligence that deals with the interaction between computers and humans in natural language."
   
]

tokens1 = [[item for item in line.split()] for line in doc_list]
g_dict1 = corpora.Dictionary(tokens1)

print("The dictionary has: " +str(len(g_dict1)) + " tokens\n")
print(g_dict1.token2id)

g_bow =[g_dict1.doc2bow(token, allow_update = True) for token in tokens1]
print("Bag of Words : ", g_bow)



print("\n--------------------------------------------------------------------------------------------------------\n")


g_dict = corpora.Dictionary([simple_preprocess(line) for line in doc_list])
g_bow = [g_dict.doc2bow(simple_preprocess(line)) for line in doc_list]

print("Dictionary : ")
for item in g_bow:
    print([[g_dict[id], freq] for id, freq in item])

g_tfidf = models.TfidfModel(g_bow, smartirs='ntc')

print("\nTF-IDF Vector:")
for item in g_tfidf[g_bow]:
    print([[g_dict[id], np.around(freq, decimals=2)] for id, freq in item])


print("\n--------------------------------------------------------------------------------------------------------\n")

'''

