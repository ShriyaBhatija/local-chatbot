"""
Implementation based on the Medium article: : https://shelwyncorte.medium.com/build-your-own-offline-ai-chatbot-running-deepseek-locally-with-ollama-d8921c20bb53 
"""

import streamlit as st
import requests
import json
from typing import Iterator, List

# Hide Streamlit menu and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""

OLLAMA_API_BASE = "http://localhost:11434"


def get_models() -> List[str]:
    try:
        with open('models.txt', 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return ["deepseek-r1:1.5b"]


def generate_stream(prompt: str, model: str) -> Iterator[str]:
    response = requests.post(
        f"{OLLAMA_API_BASE}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": True
        },
        stream=True
    )
    
    for line in response.iter_lines():
        if line:
            json_response = json.loads(line)
            if 'response' in json_response:
                yield json_response['response']
            if json_response.get('done', False):
                break

def extract_thinking_and_response(text: str) -> tuple[str, str]:
    think_start = text.find("<think>")
    think_end = text.find("</think>")
    
    if think_start != -1 and think_end != -1:
        thinking = text[think_start + 7:think_end].strip()
        response = text[think_end + 8:].strip()
        return thinking, response
    
    return "", text

def main():
    st.set_page_config(
        page_title="Local Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    with st.sidebar:
        st.title("Settings")

        available_models = get_models()
        selected_model = st.selectbox(
            "Choose a model",
            available_models,
            index=0
        )

        st.markdown(         
                """
                <div style='position: fixed; bottom: 50px; left: 30px; font-size: 13px; color: gray; text-align: left;'>
                    🚀 Developed by 
                        <a href='https://shriyabhatija.github.io' target='_blank' style='color: #1E90FF; text-decoration: none;'>
                        Shriya Bhatija</a>
                        <br>
                    12.03.2025
                </div>
                """,
                unsafe_allow_html=True
            )
    
    st.title("💭 Local Chatbot")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "thinking" in message and message["thinking"]:
                with st.expander("Show reasoning", expanded=False):
                    st.markdown(message["thinking"])
            st.markdown(message["content"])
    
    if prompt := st.chat_input("What would you like to ask?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        full_prompt = ""
        for message in st.session_state.messages:
            full_prompt += f"{message['role']}: {message['content']}\n"
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            thinking_placeholder = st.empty()
            
            message_placeholder.markdown("🤔 Thinking...")
            
            full_response = ""
            last_thinking = ""
            
            try:
                with thinking_placeholder.container():
                    thinking_expander = st.expander("Show reasoning", expanded=False)
                
                for response_chunk in generate_stream(full_prompt, selected_model):
                    full_response += response_chunk
                    thinking, response = extract_thinking_and_response(full_response)
                    
                    if thinking and thinking != last_thinking:
                        with thinking_expander:
                            st.markdown(thinking)
                        last_thinking = thinking
                    
                    message_placeholder.markdown(response + "▌")
                
                thinking, response = extract_thinking_and_response(full_response)
                message_placeholder.markdown(response)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "thinking": thinking
                })
                
            except requests.exceptions.RequestException as e:
                st.error("Error: Could not connect to Ollama server")
                st.info("Make sure Ollama is running and accessible at localhost:11434")

if __name__ == "__main__":
    main()