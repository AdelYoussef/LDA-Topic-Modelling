
from sklearn.datasets import fetch_20newsgroups
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import numpy as np





class LDA_classify :
        
        
        
    def init(self,):
        
        self.stemmer = SnowballStemmer("english")

        

    def load_models(self,lda_path, dict_path):
        
        self.load_dict(dict_path)
        self.load_lda(lda_path)
        
        
    def tokenize(self, document):
        result=[]
        for token in gensim.utils.simple_preprocess(document) :
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                Lemmatized = self.lemmatize(token)
                stemmed = self.stemm(Lemmatized)
                result.append(stemmed)     
        return result
    
    def lemmatize(self, token):
        Lemmatized = WordNetLemmatizer().lemmatize(token, pos = 'v') 
        return Lemmatized
        
    def stemm(self,token):
        stemmed =  self.stemmer.stem(token)
        return stemmed
    
    def get_topics(self,):
        self.topics = []
        for topic in self.lda_model.print_topics():
            self.topics.append(topic)
        return self.topics

    def load_lda(self, save_path):
        self.lda_model = gensim.models.ldamodel.LdaModel.load(save_path)

        
    def load_dict(self, save_path):
        self.dictionary = gensim.corpora.Dictionary.load(save_path)
        
    def classify(self,documents):
        self.get_topics()
        predictions = []
        prediction = {"document":"",
                      "topic":"",
                      "score":"",
                      "topic_content":""
                      }
        for index , document in enumerate(documents):
          processed = self.tokenize(document)
          bow_vector = self.dictionary.doc2bow(processed)
          vector =  np.array(self.lda_model[bow_vector])  
          for indexx , vec in enumerate(vector):
              prediction["document"] = index
              prediction["topic"] = vector[indexx][0]
              prediction["score"] = vector[indexx][1]
              prediction["topic_content"] = self.topics[indexx][1]
              predictions.append(prediction.copy())


        return predictions    
        
       
            
dict_path = "D:/dev_lab/LDA/saved/dict_model"
lda_path = "D:/dev_lab/LDA/saved/LDA_model"
  
test = fetch_20newsgroups(subset='test', shuffle = True)
  
mylda = LDA_classify()
mylda.init()
mylda.load_models(lda_path ,dict_path)
### examples
unseen_document = []
unseen_document.append(test.data[254])
unseen_document.append(test.data[50])
unseen_document.append(test.data[60])

predictions = mylda.classify(unseen_document)
