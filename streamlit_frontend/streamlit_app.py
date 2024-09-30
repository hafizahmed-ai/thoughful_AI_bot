import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
import requests
import uuid
import os
from dotenv import load_dotenv

load_dotenv()
# BASE_URL = 'http://localhost:8000/'
BASE_URL = 'http://rag-backend-image:8000/'

def generate_response(user_input):
    url = f"{BASE_URL}/query/"
    headers = {
        'accept': 'application/json',
        'content-type': 'application/x-www-form-urlencoded',
    }

    params = {
        'query': user_input,
    }

    response = requests.post(url, params=params, headers=headers)
    return response.json()

# Define a function for taking user-provided prompts as input
def get_text():
    input_text = st.chat_input("Ask Anything.", key="input")
    return input_text

# Initialize session states for generated responses and past inputs
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

# Generate or reuse a session UUID if it doesn't exist
if 'uuid' not in st.session_state:
    st.session_state['uuid'] = str(uuid.uuid4())
    st.sidebar.write(f"Session ID: {st.session_state['uuid']}")
else:
    st.sidebar.write(f"Session ID: {st.session_state['uuid']}")

# Create containers for input and response
response_container = st.container()
input_container = st.container()

# Styling for input box at the bottom
styl = """
<style>
    .stTextInput {
      position: fixed;
      bottom: 3rem;
    }
</style>
"""
st.markdown(styl, unsafe_allow_html=True)

# Display the user input box
with input_container:
    user_input = get_text()

# Conditional display of AI generated responses based on user prompts
with response_container:
    if user_input:
        # Generate response from the API using the current session ID (UUID)
        response = generate_response(user_input)

        # Store the user's input and AI-generated response
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            # Display the user's input as a message, with a unique key for each input
            message(st.session_state['past'][i], is_user=True, key=f'user_{i}')

            # Display the AI-generated response as a message, with a unique key for each response
            message(st.session_state["generated"][i], key=f'bot_{i}')

    # Initial greeting message if there are no interactions
    if len(st.session_state['past']) == 0:
        st.title("Greetings!")
        st.markdown("Welcome to Thoughtful AI-Bot, ready to assist. Feel free to ask any questions, and I'll do my best to provide helpful answers. Let's get started! :blush:")
