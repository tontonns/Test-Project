import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

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



st.title('Netflix\'s Movie Recommendation')
datasetPipeline = joblib.load('datasetPipeline.pkl')

def main():
    clean_dataset = datasetPipeline.named_steps['sorted_dataset']
    dropdown_value = clean_dataset['title'].tolist()
    
    user_input = st.selectbox('Choose a movie: ', dropdown_value)
    
    if st.button('Make Prediction'):
        result = make_prediction(user_input)
        result.reset_index(inplace=True)
        result.index = result.index + 1
        result.drop(columns=['index', 'score'], inplace=True)
        
        st.success(f'Recommendation result:')
        st.dataframe(result)
    
    return None

def make_prediction(user_input):
    clean_dataset = datasetPipeline.named_steps['clean_dataset']
    selected_features = datasetPipeline.named_steps['selected_features']
    
    model_handler = ModelHandler(clean_dataset, selected_features)
    model_handler.compute_similarity_matrix()
    
    recommendations = model_handler.recommend(user_input)
    return recommendations

if __name__ == '__main__':
    main()