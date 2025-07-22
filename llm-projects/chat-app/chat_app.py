# Local LLM Chat Application
# Requirements: streamlit, requests, fastapi, uvicorn, ollama

import streamlit as st
import requests
import json
from datetime import datetime
import time

def main():
    st.set_page_config(
        page_title="Local LLM Chat",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Local LLM Chat Application")
    st.markdown("Chat with local LLMs using Ollama")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Backend URL
        backend_url = st.text_input(
            "Backend URL", 
            value="http://localhost:8000",
            help="URL of the FastAPI backend"
        )
        
        # Model selection
        if st.button("Refresh Models"):
            try:
                response = requests.get(f"{backend_url}/models")
                if response.status_code == 200:
                    st.session_state.available_models = response.json()["models"]
                else:
                    st.error("Failed to fetch models")
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")
        
        # Initialize available models
        if "available_models" not in st.session_state:
            st.session_state.available_models = ["llama2", "mistral", "codellama", "gemma3"]
        
        selected_model = st.selectbox(
            "Select Model",
            options=st.session_state.available_models,
            index=0
        )
        
        # Chat settings
        st.subheader("Chat Settings")
        max_messages = st.slider("Max Messages in Context", 5, 50, 20)
        
        # Clear chat button
        if st.button("Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "timestamp" in message:
                    st.caption(f"*{message['timestamp']}*")
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "timestamp": timestamp
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(f"*{timestamp}*")
        
        # Generate assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                # Prepare messages for API (limit context)
                recent_messages = st.session_state.messages[-max_messages:]
                api_messages = [
                    {"role": msg["role"], "content": msg["content"]} 
                    for msg in recent_messages
                ]
                
                # Show typing indicator
                with st.spinner("Thinking..."):
                    response = requests.post(
                        f"{backend_url}/chat",
                        json={
                            "messages": api_messages,
                            "model": selected_model
                        },
                        timeout=60
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    assistant_response = result["response"]
                    response_timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    # Display response
                    message_placeholder.markdown(assistant_response)
                    st.caption(f"*{response_timestamp} • Model: {selected_model}*")
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_response,
                        "timestamp": response_timestamp,
                        "model": selected_model
                    })
                else:
                    error_msg = f"Error: {response.status_code} - {response.text}"
                    message_placeholder.error(error_msg)
                    
            except requests.exceptions.RequestException as e:
                error_msg = f"Connection error: {str(e)}\n\nMake sure the FastAPI backend is running at {backend_url}"
                message_placeholder.error(error_msg)
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                message_placeholder.error(error_msg)
    
    # Instructions
    with st.expander("📋 Setup Instructions"):
        st.markdown("""
        ### Setup Steps:
        
        1. **Install Dependencies:**
        ```bash
        pip install streamlit fastapi uvicorn ollama requests
        ```
        
        2. **Install Ollama:**
        - Visit https://ollama.ai and install Ollama
        - Pull models: `ollama pull llama2` or `ollama pull mistral`
        
        3. **Run Backend:**
        ```bash
        python backend.py
        ```
        
        4. **Run Streamlit:**
        ```bash
        streamlit run chat_app.py
        ```
        
        ### Available Models:
        - llama2 (7B, 13B, 70B variants)
        - mistral (7B)
        - codellama (for coding tasks)
        - gemma3 (Google's model)
        """)

if __name__ == "__main__":
    main()