# app.py

import streamlit as st
import os
import time
import re
import json
import pandas as pd
from dotenv import load_dotenv
from src import ChatwithCSV
import asyncio
from src import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

# Load environment variables
load_dotenv()

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("⚠️ OPENAI_API_KEY not found in .env file. Please create a .env file with your OpenAI API key.")
    st.stop()

st.set_page_config(page_title="Interactive CSV Q&A Chatbot", layout="wide")

st.title("Interactive CSV Q&A Chatbot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "csv_uploaded" not in st.session_state:
    st.session_state.csv_uploaded = False
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "df" not in st.session_state:
    # Load default CSV file
    default_csv_path = os.path.join("src", "data", "titanic.csv")
    try:
        st.session_state.df = pd.read_csv(default_csv_path)
        st.session_state.csv_uploaded = True
        st.session_state.default_csv_loaded = True
        logger.info(f"Loaded default CSV file: {default_csv_path}")
    except Exception as e:
        st.error(f"Error loading default CSV file: {e}")
        logger.error(f"Error loading default CSV: {e}")
        st.session_state.csv_uploaded = False
        st.session_state.default_csv_loaded = False

# Sidebar for configuration
with st.sidebar:
    st.header("Upload and Configure")
    
    st.info("🔑 Using OpenAI API Key from .env file")
    st.info("🤖 Using Langchain Agent")

    st.markdown("---")

    # Display default CSV info
    if st.session_state.get("default_csv_loaded", False):
        st.success("📊 Using default CSV: **titanic.csv**")
        st.info(f"Dataset loaded: {len(st.session_state.df)} rows, {len(st.session_state.df.columns)} columns")
    
    # CSV File Uploader (optional - allows overriding default)
    st.subheader("Upload Custom CSV File (Optional)")
    uploaded_file = st.file_uploader("Choose a CSV file to replace default", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.csv_uploaded = True
            st.session_state.default_csv_loaded = False
            st.success(f"File `{uploaded_file.name}` uploaded successfully and will be used instead of default.")
            logger.info(f"Uploaded file: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error reading the CSV file: {e}")
            logger.error(f"Error reading CSV: {e}")

    if st.session_state.csv_uploaded:
        st.info("✅ CSV file is ready for querying.")
    
    # Display DataFrame preview
    st.markdown("---")
    st.subheader("📋 Dataset Preview")
    if st.session_state.csv_uploaded and "df" in st.session_state:
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        st.caption(f"Showing first 10 rows of {len(st.session_state.df)} total rows")

# Define tabs
tab_chat, tab_faqs, tab_samples, tab_contact = st.tabs(["Chat", "FAQs", "Sample Queries", "📞 Contact Me"])

# Asynchronous function to initialize the chatbot
async def initialize_chatbot():
    if st.session_state.chatbot is None:
        df = st.session_state.df
        chatbot = ChatwithCSV(api_key=OPENAI_API_KEY, df=df)
        st.session_state.chatbot = chatbot
        logger.info("Chatbot initialized successfully with OpenAI and Langchain agent.")

# Chat tab
with tab_chat:
    if not st.session_state.csv_uploaded:
        st.warning("⚠️ CSV file not loaded. Please check the sidebar for errors.")
    else:
        if st.session_state.chatbot is None:
            asyncio.run(initialize_chatbot())

        if st.session_state.chatbot:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    content = message["content"]
                    # Handle both old format (string) and new format (dict)
                    if isinstance(content, dict):
                        st.markdown(content.get("answer", ""))
                        # Show query details for assistant messages
                        if message["role"] == "assistant":
                            query_executed = content.get("query_executed")
                            query_output = content.get("query_output")
                            if query_executed:
                                with st.expander("🔍 View Query Executed", expanded=False):
                                    st.code(query_executed, language="python")
                            if query_output:
                                with st.expander("📊 View Query Output", expanded=False):
                                    st.text(str(query_output))
                    else:
                        # Old format - just display the string
                        st.markdown(content)

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

            async def handle_user_input(prompt: str):
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
                            result = await st.session_state.chatbot.chat_with_a_df(prompt)
                            # Handle both dict (new format) and string (old format) for backward compatibility
                            if isinstance(result, dict):
                                answer = result.get("answer", "I don't know")
                                query_executed = result.get("query_executed")
                                query_output = result.get("query_output")
                            else:
                                # Fallback for old format
                                answer = result
                                query_executed = None
                                query_output = None
                        except Exception as e:
                            st.error(f"Error processing your request: {e}")
                            answer = "I'm sorry, I couldn't process your request."
                            query_executed = None
                            query_output = None

                    # Store message with query details
                    message_content = {
                        "answer": answer,
                        "query_executed": query_executed,
                        "query_output": query_output
                    }
                    st.session_state.messages.append({"role": "assistant", "content": message_content})

                    with st.chat_message("assistant"):
                        # Display the answer
                        message_placeholder = st.empty()
                        full_response = ""
                        for word in response_generator(answer):
                            full_response += word
                            message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)
                        
                        # Display query executed and output in expandable sections
                        if query_executed:
                            with st.expander("🔍 View Query Executed", expanded=False):
                                st.code(query_executed, language="python")
                        
                        if query_output:
                            with st.expander("📊 View Query Output", expanded=False):
                                # Try to display as dataframe if it looks like tabular data
                                try:
                                    # Check if output looks like a pandas Series or DataFrame string representation
                                    if isinstance(query_output, str) and ("\n" in query_output or "Name:" in query_output):
                                        st.text(query_output)
                                    else:
                                        st.text(str(query_output))
                                except:
                                    st.text(str(query_output))
                        
                        logger.info(f"User prompt: {prompt} | Response: {answer}")

            prompt = st.chat_input(placeholder="Ask me anything about your CSV data...")
            if prompt:
                asyncio.run(handle_user_input(prompt))

# FAQs tab
with tab_faqs:
    st.header("FAQs")
    st.markdown("Here are some frequently asked questions:")

    faqs = [
        {"question": "What CSV file is being used?", "answer": "The app uses a default CSV file (titanic.csv) located at src/data/titanic.csv. You can upload your own CSV file from the sidebar to replace the default."},
        {"question": "How do I upload my own CSV file?", "answer": "Use the optional file uploader in the sidebar to upload your CSV file. It will replace the default titanic.csv dataset."},
        {"question": "How do I ask a question about the CSV data?", "answer": "Type your question in the chat input at the bottom of the Chat tab."},
        {"question": "How do I set up the OpenAI API Key?", "answer": "Create a .env file in the project root directory and add your OpenAI API key: OPENAI_API_KEY=your_api_key_here. The app will automatically load it when you start."},
        {"question": "How do I create an OpenAI API Key?", "answer": "To create an OpenAI API Key, visit the [OpenAI API Keys Page](https://platform.openai.com/settings/organization/api-keys). Log in to your OpenAI account, click on 'Create new secret key', and then add it to your .env file."},
        {"question": "What agent is being used?", "answer": "The app uses Langchain agent with OpenAI's GPT-4o-mini model for processing queries and analyzing CSV data."},
    ]

    for faq in faqs:
        with st.expander(f"Q: {faq['question']}"):
            st.write(f"A: {faq['answer']}")

# Sample Queries tab
with tab_samples:
    st.header("Sample Queries")

    json_file_path = os.path.join("src", "constants", "sample_queries.json")  # Corrected the path separator for cross-platform compatibility

    # Load sample queries only once
    if "sample_queries" not in st.session_state:
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    st.session_state.sample_queries = json.load(f)
                logger.info("Loaded sample queries successfully.")
            except json.JSONDecodeError:
                st.error("Invalid JSON file. Please check the file format.")
                logger.error("Invalid JSON format in sample_queries.json.")
                st.session_state.sample_queries = []
        else:
            st.warning(f"JSON file not found at path: {json_file_path}")
            logger.warning(f"Sample queries JSON file not found at: {json_file_path}")
            st.session_state.sample_queries = []
    
    sample_queries = st.session_state.get("sample_queries", [])

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
