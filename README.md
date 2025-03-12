# Local chatbot using Ollama and Streamlit
This project builds an offline AI chatbot using Ollama and Streamlit, with deep learning models for natural language processing. The chatbot can have context-aware conversations by maintaining the session history, allowing for more coherent and intelligent responses. Models available include leightweight versions DeepSeek-R1, Llama 3.2 and Gemma 3 are available. 

## Requirements
Download and install Ollama from [ollama.com/download](https://ollama.com/download)

Install the models:

``` 
ollama run deepseek-r1:1.5b
ollama run gemma3:1b
ollama run llama3.2:1b 
```

Create a virtual environment and install the dependencies:
```
pip install -r requirements.txt
```

## Running the chatbot
Type the following command to use the chatbot:
```
streamlit run chatbot.py
```

The chatbot will be available at ```http://localhost:8501```