import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
model = pickle.load(open('title_from_index.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    movies_data=pd.read_csv("movies (1).csv")
    selected_features= ['genres','keywords','tagline','cast','director']
    #replacing null values
    for features in selected_features:
        movies_data[features]= movies_data[features].fillna('')
    #combining all the 5 in selected feature
    combined_feature=movies_data['genres']+movies_data['keywords']+movies_data['tagline']+movies_data['cast']+movies_data['director']
    #converting the text data to feature vectors
    vectorizer =TfidfVectorizer()
    feature_vector=vectorizer.fit_transform(combined_feature)
    #cosine similarity  
    similarity=cosine_similarity(feature_vector)
    # getting movie name from user
    movie_name=list(request.form.values())[0]
    #creating a list with all the movie names given in the dataset
    list_of_all_titles=(movies_data['title'].tolist())
    #finding the close match for the movie name given by the user
    from difflib import get_close_matches
    find_close_match =get_close_matches(movie_name,list_of_all_titles)
    find_close_match
    close_match=find_close_match[0]
    #finding index with title
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    #getting list of similar movies
    similarity_score= list(enumerate(similarity[index_of_the_movie]))
    similarity[index_of_the_movie]
    #sorting the movies based on their similarity score
    sorted_similar_movies=sorted(similarity_score,key=lambda x:x[1],reverse=True)
    #print movie name similar to entereed input
    i=1
    recommend_movies=list()
    for movies in sorted_similar_movies:
        index=movies[0]
        title_from_index=movies_data[movies_data.index==index]['title'].values[0]
        if (i<=20):
            #print(i,'',title_from_index)
            recommend_movies.append( title_from_index)
            i+=1

    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
   # Movie_name= request.form.values()[0]
   # prediction = title_from_index.predict(final_features)

    output = recommend_movies

    return render_template('index.html', prediction_text='Movies are {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)