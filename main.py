import streamlit as st
import os
import time
import re
import json
import pandas as pd
from dotenv import load_dotenv
from src import ChatwithCSV  # Ensure this class is in a separate file named ChatwithCSV.py

# Load environment variables (optional, since we're accepting the API key via input)
load_dotenv()

st.title("Interactive CSV Q&A Chatbot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "csv_uploaded" not in st.session_state:
    st.session_state.csv_uploaded = False
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None

# Sidebar for CSV upload and OpenAI API key input
with st.sidebar:
    st.header("Upload and Configure")

    # Input for OpenAI API Key
    st.subheader("OpenAI API Key")
    openai_api_key_input = st.text_input(
        "Enter your OpenAI API Key",
        type="password",
        help="Your API key will be used to interact with OpenAI's services.",
    )
    if st.button("Set API Key"):
        if openai_api_key_input:
            st.session_state.openai_api_key = openai_api_key_input
            st.success("OpenAI API Key has been set.")
        else:
            st.error("Please enter a valid OpenAI API Key.")
    
    # Added URL and instructions below the API key button
    st.markdown("""
    ---
    ### How to Create an OpenAI API Key
    To create an OpenAI API Key, follow these steps:
    1. Visit the [OpenAI API Keys Page](https://platform.openai.com/settings/organization/api-keys).
    2. Log in to your OpenAI account. If you don't have one, you'll need to create it.
    3. Click on "Create new secret key".
    4. Copy the generated API key and paste it into the input field above.
    """)

    st.markdown("---")

    # CSV File Uploader
    st.subheader("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.csv_uploaded = True
            st.success(f"File `{uploaded_file.name}` uploaded successfully.")
        except Exception as e:
            st.error(f"Error reading the CSV file: {e}")

    if st.session_state.csv_uploaded:
        st.info("CSV file is ready for querying.")

# Define tabs
tab_chat, tab_faqs, tab_samples, tab_contact = st.tabs(["Chat", "FAQs", "Sample Queries", "📞 Contact Me"])

# Chat tab
with tab_chat:
    if not st.session_state.csv_uploaded:
        st.warning("Please upload a CSV file from the sidebar to start chatting.")
    elif not getattr(st.session_state, 'openai_api_key', None):
        st.warning("Please enter your OpenAI API Key in the sidebar to start chatting.")
    else:
        # Initialize ChatwithCSV if not already done
        if st.session_state.chatbot is None:
            st.session_state.chatbot = ChatwithCSV(st.session_state.openai_api_key)

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        def response_generator(response):
            for word in response.split():
                yield word + " "
                time.sleep(0.05)

        def normalize_text(text):
            return re.sub(r'[^\w\s]', '', text).lower().strip()

        greeting_responses = {
            "hi": "Hello! How can I assist you today?",
            "hello": "Hello! What can I do for you?",
            "hey": "Hey there! How can I help?",
            "good morning": "Good morning! What can I assist you with?",
            "good afternoon": "Good afternoon! How can I help you today?",
            "good evening": "Good evening! How may I assist you?"
        }

        if prompt := st.chat_input(placeholder="Ask me anything about your CSV data..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            normalized_prompt = normalize_text(prompt)

            if normalized_prompt in greeting_responses:
                response = greeting_responses[normalized_prompt]
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
            else:
                with st.spinner("Thinking..."):
                    try:
                        answer = st.session_state.chatbot.chat_with_a_df(st.session_state.df, prompt)
                    except Exception as e:
                        st.error(f"Error processing your request: {e}")
                        answer = "I'm sorry, I couldn't process your request."

                st.session_state.messages.append({"role": "assistant", "content": answer})

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    for word in response_generator(answer):
                        full_response += word
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)

# FAQs tab
with tab_faqs:
    st.header("FAQs")
    st.markdown("Here are some frequently asked questions:")

    faqs = [
        {"question": "How do I upload a CSV file?", "answer": "Use the file uploader in the sidebar to upload your CSV file."},
        {"question": "How do I ask a question about the CSV data?", "answer": "Type your question in the chat input at the bottom of the Chat tab."},
        {"question": "How do I change the OpenAI API Key?", "answer": "Enter a new API key in the OpenAI API Key section in the sidebar and click 'Set API Key'."},
        {"question": "Why do I need an OpenAI API Key?", "answer": "The API key is required to interact with OpenAI's language models for generating responses."},
        # Added FAQ for creating OpenAI API Key
        {"question": "How do I create an OpenAI API Key?", "answer": "To create an OpenAI API Key, visit the [OpenAI API Keys Page](https://platform.openai.com/settings/organization/api-keys). Log in to your OpenAI account, click on 'Create new secret key', and then copy the generated key into the OpenAI API Key field in the sidebar."},
    ]

    for faq in faqs:
        with st.expander(f"Q: {faq['question']}"):
            st.write(f"A: {faq['answer']}")

# Sample Queries tab
with tab_samples:
    st.header("Sample Queries")

    json_file_path = "src/constants/sample_queries.json"  # Corrected the path separator for cross-platform compatibility

    if os.path.exists(json_file_path):
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                sample_queries = json.load(f)
        except json.JSONDecodeError:
            st.error("Invalid JSON file. Please check the file format.")
            sample_queries = []
    else:
        st.warning(f"JSON file not found at path: {json_file_path}")
        sample_queries = []

    csv_url = "https://example.com/sample.csv"  # Replace with your actual CSV URL
    st.markdown(f"[Download Sample CSV]({csv_url})")

    if sample_queries:
        st.markdown("Here are some sample queries and their answers created from the above CSV:")

        for query in sample_queries:
            st.subheader(f"Q: {query['question']}")
            st.write(f"A: {query['answer']}")
    else:
        st.info("No sample queries to display. Please ensure the JSON file path is correct and the file is properly formatted.")

# Contact tab
with tab_contact:
    st.header("📞 Contact Information")
    st.write("Feel free to reach out through any of the following platforms 😊: ")

    st.markdown("**📧 Email**")
    if st.button("pwaykos1@gmail.com"):
        st.write("mailto:pwaykos1@gmail.com")

    st.markdown("**📱 Phone**")
    if st.button("7249542810"):
        st.write("tel:+17249542810")

    st.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/prajwal-waykos/)")
    st.markdown("**[🗃️ Resume](https://drive.google.com/file/d/1OiSCu4e_1R7cawKSU80cr63Cd2-4OVq7/view?usp=drivesdk)**")
    st.markdown("**[🐙 GitHub](https://github.com/praj-17)**")
