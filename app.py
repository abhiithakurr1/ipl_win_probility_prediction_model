import streamlit as st
import pickle
import pandas as pd

# Team and city lists
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# Load model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# App title
st.title('🏏 IPL Win Predictor')

# Team selection
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the Bowling Team', sorted(teams))

# Check if same teams are selected
if batting_team == bowling_team:
    st.warning("Batting and Bowling teams must be different!")

# City selection
selected_city = st.selectbox('Select Host City', sorted(cities))

# Target input
target = st.number_input('Target Score', min_value=1)

# Match progress
col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Current Score', min_value=0)
with col4:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, step=0.1, value=5.0)
with col5:
    wickets_out = st.number_input('Wickets Fallen', min_value=0, max_value=10)

# Predict button
if st.button('Predict Probability'):

    if overs == 0:
        st.error("Overs completed can't be 0.")
    elif batting_team == bowling_team:
        st.warning("Select different teams for batting and bowling.")
    else:
        try:
            runs_left = target - score
            balls_left = 120 - int(overs * 6)
            wickets = 10 - int(wickets_out)
            crr = score / overs
            rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

            input_df = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [selected_city],
                'runs_left': [runs_left],
                'balls_left': [balls_left],
                'wickets': [wickets],
                'total_runs_x': [target],
                'crr': [crr],
                'rrr': [rrr]
            })

            # Predict
            result = pipe.predict_proba(input_df)
            loss = result[0][0]
            win = result[0][1]

            # Display result
            st.subheader(f"🏏 {batting_team}: **{round(win * 100)}%** chance to win")
            st.subheader(f"🎯 {bowling_team}: **{round(loss * 100)}%** chance to win")

        except Exception as e:
            st.error(f"Something went wrong during prediction: {e}")
