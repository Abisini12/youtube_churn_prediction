# Import libraries
import streamlit as st
import numpy as np
import pickle
import os

model_path = 'logisticfinal_model.pkl' # Update this to the correct path if needed
loaded_model = pickle.load(open(model_path, 'rb'))

# Function to preprocess input data and make predictions
def churn_prediction(input_data):
    # Map 'Gender' and 'Genre' to numerical values using one-hot encoding
    gender_mapping = {'Male': 0, 'Female': 1}
    genre_mapping = {'Sports': 0, 'Series': 1, 'Education': 2, 'Entertainment': 3}  # Update with your genre categories

    # Preprocess input data
    gender = gender_mapping.get(input_data[0], 0)  # Default to 0 if not found in mapping
    genre = genre_mapping.get(input_data[1], 0)  # Default to 0 if not found in mapping
    num_of_videos = float(input_data[2])
    like_dislike = float(input_data[3])
    is_active_member = float(input_data[4])
    age = float(input_data[5])
    streamed_time = float(input_data[6])

    # Create a NumPy array with the preprocessed data
    input_data_as_numpy_array = np.array([gender, genre, num_of_videos, like_dislike, is_active_member, age, streamed_time])

    # Reshape the data and make the prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The subscriber is predicted as not leaving the channel..'
    else:
        return 'The subscriber is predicted to unsubscribe the channel..'

# Rest of your code remains the same...
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
