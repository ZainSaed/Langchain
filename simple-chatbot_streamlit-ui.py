from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# Page Title
st.title("🤖 AI Chatbot")

# Input Box
user_input = st.text_input("Ask Anything")

# Button
if st.button("Send"):

    if user_input:

        # Get Response
        response = model.invoke(user_input)

        # Display Response
        st.write("### AI Response")
        st.write(response.content)

    else:
        st.warning("Please enter a question.")