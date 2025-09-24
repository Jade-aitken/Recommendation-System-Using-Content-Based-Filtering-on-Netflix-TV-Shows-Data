import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt           
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        
    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        
    def remove_outliers(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)] 
    
    def remove_values(self, column, val):
        self.df = self.df[~self.df[column].isin(val)]
        
    def fill_NaN(self, column, val):
        self.df[column] = self.df[column].fillna(val)
    
    def drop_column(self, column):
        self.df.drop(columns=[column],inplace=True)
    

class ModelHandler:
    def __init__(self,df):
        self.data = df
        self.newdata = None
        self.tfidf = None
        self.tfidf_matrix = None
        self.cosine_sim = None
    
    def combined_features(self):
        def combine(row):
            return ' '.join([
                row['director'],
                row['cast'],
                row['country'],
                str(row['release_year']),
                row['rating'],
                row['duration'],
                row['listed_in'],
                row['description']
            ])
        self.newdata = self.data.copy()
        self.newdata['content_features'] = self.newdata.apply(combine, axis=1)
        
    
    def train_model(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.newdata['content_features'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
    
    def recommend(self, title, top_n=5):
        indices = pd.Series(self.newdata.index, index=self.newdata['title']).drop_duplicates()
        if title not in indices:
            return f"'{title}' tidak ditemukan di data."

        idx = indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        rec_indices = [i[0] for i in sim_scores]

        return self.newdata.iloc[rec_indices]
    
    def save_data(self,filename):
        with open(filename, 'wb') as f:
            pkl.dump(self.newdata, f)
    
    def save_tfidf_vector(self,filename):
        with open(filename, 'wb') as f:
            pkl.dump(self.tfidf, f)
            
    def save_tfidf_matrix(self,filename):
        with open(filename, 'wb') as f:
            pkl.dump(self.tfidf_matrix, f)
    
    def save_cosine_sim(self,filename):
        with open(filename, 'wb') as f:
            pkl.dump(self.cosine_sim, f)
            


data_handler = DataHandler('netflix_titles.csv')
data_handler.load_data()
data_handler.remove_outliers('release_year')
data_handler.remove_values('rating',['74 min','84 min','66 min'])
data_handler.fill_NaN('director','Unknown')
data_handler.fill_NaN('cast','Unknown')  
data_handler.fill_NaN('country','Unknown')
data_handler.fill_NaN('rating','TV-MA')
data_handler.drop_column('show_id')
data_handler.drop_column('date_added')
data_handler.remove_values('type',['Movie'])
data_handler.drop_column('type')  

X = data_handler.df

model_handler = ModelHandler(X)
model_handler.combined_features()
model_handler.train_model()
recommendations = model_handler.recommend("Blood & Water")
print(recommendations)

model_handler.save_data('df.pkl')
#model_handler.save_tfidf_vector('tfidf_vectorizer.pkl')
model_handler.save_tfidf_matrix('tfidf_matrix.pkl') 
#model_handler.save_cosine_sim('cosine_similarity.pkl')   