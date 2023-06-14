
from sklearn.datasets import fetch_20newsgroups
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import numpy as np
np.random.seed(619)





class LDA :
        
        
        
    def init(self,):
        
        #incase needed to redownload
        
        # nltk.download('omw-1.4') 
        # nltk.download('wordnet')
        self.stemmer = SnowballStemmer("english")
            
    
    def load_data(self,train_data):
        
        self.train_data = train_data
        return self.train_data
    
    def preprocess(self, documents):
        
        processed_docs = []
    
        for document in documents:
            processed_docs.append(self.tokenize(document))
            
        return processed_docs 


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
    
    
    def doc2bow_dict(self,processed_docs):
        self.dictionary = gensim.corpora.Dictionary(processed_docs)
        self.bag_of_words = [self.dictionary.doc2bow(doc) for doc in processed_docs]
        self.dictionary.save("D:/dev_lab/LDA/saved/dict_model")
        print("dict saved successfully")

        return self.bag_of_words, self.dictionary
    

    def filter_docs(self,min_reps , doc_occur , keep_most_freq = None):


        self.dictionary.filter_extremes(no_below = min_reps, no_above = doc_occur , keep_n= keep_most_freq)
        
        
        
    def LDA_model (self,num_topics , dictionary, epochs, cores=2):
        
        if self.multicore == True:
            print("training begins ++++++ using multicore \n")
            self.lda_model =  gensim.models.LdaMulticore(self.bag_of_words, num_topics = num_topics, id2word = dictionary, passes = epochs, workers = cores)
            
        elif self.multicore == False:
            print("training begins ++++++ using singlecore \n")
            self.lda_model = gensim.models.LdaModel(self.bag_of_words, num_topics = num_topics, id2word = dictionary, passes = epochs)
            
        return self.lda_model
        
    def get_topics(self,):
        self.topics = []
        for topic in self.lda_model.print_topics():
            self.topics.append(topic)
        return self.topics


    def train(self, save_path = "", save_model = True , Filter = True , multicore = True , cores = 2 , batch_size = 10 , epochs = 50 ,  num_topics=10 ):
        self.multicore = multicore
        self.Filter = Filter
        self.processed_docs = self.preprocess(self.train_data)
        
        self.doc2bow_dict(self.processed_docs)
        
        
        if self.Filter == True:
            self.filter_docs(min_reps = 15, min_occr = 0.1, keep_most_freq = None)
        
        
        self.LDA_model(num_topics = num_topics, dictionary = self.dictionary, epochs = epochs)
        print("training successful \n")

        if save_model == True:
            self.lda_model.save(save_path + "LDA_model")
            print("model saved successfully")
            
    

                
            

  
train = fetch_20newsgroups(subset='train', shuffle = True)
  
mylda = LDA()
mylda.init()
mylda.load_data(train.data)
mylda.train(Filter = False , save_path = "D:/dev_lab/LDA/saved/" , multicore = False , batch_size = 10 , epochs = 200 ,num_topics=10)


