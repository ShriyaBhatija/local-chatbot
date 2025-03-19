"""
Implementation based on the Medium article: : https://shelwyncorte.medium.com/build-your-own-offline-ai-chatbot-running-deepseek-locally-with-ollama-d8921c20bb53 
"""

import streamlit as st
import requests
import json
from typing import Iterator, List

# hide Streamlit menu and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""

# Ollama API is hosted on localhost at port 11434
# base URL for Ollama API
OLLAMA_API_BASE = "http://localhost:11434"

# get models from models.txt file
def get_models() -> List[str]:
    try:
        with open('models.txt', 'r') as f:
            models = [line.strip() for line in f if line.strip()]
            if not models:
                return ["deepseek-r1:1.5b"]
    except FileNotFoundError:
        return ["deepseek-r1:1.5b"]


def generate_stream(prompt: str, model: str) -> Iterator[str]:
    # sends a POST request to the Ollama server to generate a response based on the provided model and prompt
    response = requests.post(
        f"{OLLAMA_API_BASE}/api/generate", #/api/generate endpoint is used to generate a response/completion for a given prompt
        json={
            "model": model,
            "prompt": prompt,
            "stream": True  # stream the response
        },
        stream=True # stream the response: response will be received in chunks, allowing the client to process it incrementally as it arrives
    )
    
    # process the streamed response from the Ollama server
    for line in response.iter_lines():
        if line:
            json_response = json.loads(line)
            if 'response' in json_response:
                yield json_response['response']
            if json_response.get('done', False):
                break

# extract the "thinking" and "response" parts from a given text
def extract_thinking_and_response(text: str) -> tuple[str, str]:
    think_start = text.find("<think>")
    think_end = text.find("</think>")
    
    if think_start != -1 and think_end != -1:
        thinking = text[think_start + 7:think_end].strip()
        response = text[think_end + 8:].strip()
        return thinking, response
    
    return "", text

def main():
    # configure the settings for the Streamlit app page
    st.set_page_config(
        page_title="Local Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    # allow to render Markdown text
    # allow to apply custom CSS and HTML styles in the Streamlit app
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # Sidebar for model selection
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
    
    # st.session_state is a special dictionary provided by Streamlit to store variables that need to persist across different interactions within the same session
    if "messages" not in st.session_state:
        st.session_state.messages = [] # list that stores the conversation history between the user and the chatbot
    
    # iterate through the messages stored in the Streamlit session state and display them in the chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "thinking" in message and message["thinking"]:
                with st.expander("Show reasoning", expanded=False):
                    st.markdown(message["thinking"])
            st.markdown(message["content"])
    

    #  handle user input in the Streamlit app and updates the chat interface accordingly
    if prompt := st.chat_input("What would you like to ask?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): # create a chat message block in the Streamlit app for the user's input
            st.markdown(prompt)

        # add memory
        full_prompt = ""
        for message in st.session_state.messages:
            full_prompt += f"{message['role']}: {message['content']}\n"
        
        # generate and display of the assistant's response in the chat interface
        with st.chat_message("assistant"): # chat message block in the Streamlit app for the assistant's response
            message_placeholder = st.empty() # empty placeholders
            thinking_placeholder = st.empty() # empty placeholders
            
            message_placeholder.markdown("🤔 Thinking...") # display a "Thinking..." message
            
            full_response = ""
            last_thinking = ""
            
            try:
                with thinking_placeholder.container(): # container within thinking_placeholder
                    thinking_expander = st.expander("Show reasoning", expanded=False) # expander widget labeled "Show reasoning" that is initially collapsed
                
                for response_chunk in generate_stream(full_prompt, selected_model):
                    full_response += response_chunk
                    thinking, response = extract_thinking_and_response(full_response)
                    
                    if thinking and thinking != last_thinking: # checks if there is new thinking content that has not been displayed yet
                        with thinking_expander:
                            st.markdown(thinking)
                        last_thinking = thinking
                    
                    message_placeholder.markdown(response + "▌") # updates the message_placeholder with the current response and a cursor (▌) to indicate that the response is still being generated
                
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