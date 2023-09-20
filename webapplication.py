# Import libraries
import streamlit as st
import numpy as np
import pickle
import io
import requests
import os

# Define the URL of the raw pickle file in your GitHub repository
github_raw_url = 'https://github.com/Abisini12/youtube_churn_prediction/blob/0f765108e8feb101efa920cbe4f23455c1377ea3/logisticfinal_model.pkl'

# Fetch the raw content of the pickle file from GitHub
response = requests.get(github_raw_url)

if response.status_code == 200:
    # Load the pickle content from the response
    pickle_content = response.content
    
    # Deserialize the pickle content into the model
    loaded_model = pickle.load(pickle_content)
    
    # Now, you can use the loaded model as needed
else:
    print("Failed to fetch the pickle file from GitHub")

# Create a function for prediction
def churn_prediction(input_data):
    
    input_data_as_numpy_array = np.asarray(input_data).astype(float)  # Convert input data to float
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return 'The subscriber is predicted as not leaving the channel..'
    else:
        return 'The subscriber is predicted to unsubscribe the channel..'

def main():
    # Setting Application title
    st.title('Youtube Subscriber Churn Prediction')

    Gender = st.text_input('Enter the gender:')
    Genre = st.text_input('Enter the genre:')
    NumOfVideo = st.text_input('The number of videos watched:')
    Like_Dislike = st.text_input('Whether the video is liked(1) or disliked(0):')
    IsActiveMember = st.text_input('Is the subscriber active(1) or not(0):')  # Corrected the variable name here
    Age = st.text_input('Enter the age:')
    Streamedtime = st.text_input('Enter the streamed time of the subscriber:')

    # Code for prediction
    diagnosis = ''
    
    # Creating a button for prediction
    if st.button('Predict'):
        diagnosis = churn_prediction([Gender, Genre, NumOfVideo, Like_Dislike, IsActiveMember, Age, Streamedtime])
    st.success(diagnosis)

if __name__ == '__main__':
    main()


