import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.pipeline import Pipeline
import pickle as pkl

class DataHandler:
  def __init__(self, file_path):
    self.file_path = file_path
    self.dataset = None
    self.selected_features = None
    self.sorted_dataset = None

  def load_data(self):
    self.dataset = pd.read_csv(self.file_path)

  def clean_data(self):
    self.dataset['date_added'] = pd.to_datetime(self.dataset['date_added'], errors='coerce').dt.date
    self.dataset.drop(columns=['show_id'], inplace=True)

    self.invalid_ratings = ['74 min', '84 min', '66 min']
    self.dataset = self.dataset[self.dataset['rating'].isin(self.invalid_ratings) == False]

    self.dataset.drop(columns=['director', 'country', 'date_added', 'duration'], inplace=True)

    self.dataset.dropna(inplace=True)
    self.dataset.drop_duplicates(inplace=True)
    self.dataset.reset_index(drop=True, inplace=True)

  def generate_selected_features(self):
    self.dataset['selected_features'] = self.dataset['listed_in'] + ' ' + self.dataset['title'] + ' ' + self.dataset['rating'] + ' ' + self.dataset['description']
    self.selected_features = self.dataset['selected_features']

  def sort_by_title(self):
    self.sorted_dataset = self.dataset.sort_values(by='title')

  def dump_pickle(self):
    pipeline = Pipeline([
        ('clean_dataset', self.sorted_dataset),
        ('selected_features', self.selected_features)
    ])
    pkl.dump(pipeline, open('datasetPipeline.pkl', 'wb'))
    

class ModelHandler:
  def __init__(self, dataset, selected_features):
    self.dataset = dataset
    self.selected_features = selected_features
    self.tfidf = TfidfVectorizer(stop_words='english')
    self.tfidf_matrix = None
    self.cosine_sim = None

  def compute_similarity_matrix(self):
    self.tfidf_matrix = self.tfidf.fit_transform(self.selected_features)
    self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

  def recommend(self, title, num_recommend=5):
    indices = pd.Series(self.dataset.index, index=self.dataset['title'])

    if title not in indices:
      raise ValueError(f"Title '{title}' not found in the dataset.")

    index = indices[title]
    sim_scores = list(enumerate(self.cosine_sim[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_similar = sim_scores[1:num_recommend+1]
    netflix_indices = [i[0] for i in top_similar]

    return_data = pd.DataFrame(self.dataset[['title', 'type', 'cast', 'rating', 'listed_in']].iloc[netflix_indices])
    return_data['score'] = np.array(top_similar)[:,1]
    return return_data

file_path = 'netflix_titles.csv'

data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.clean_data()
data_handler.generate_selected_features()
data_handler.sort_by_title()
data_handler.dump_pickle()

model_handler = ModelHandler(data_handler.dataset, data_handler.selected_features)
model_handler.compute_similarity_matrix()

recommendations = model_handler.recommend('Transformers: War for Cybertron: Kingdom')
recommendations