# Recipe Chatbot

A chatbot that uses a Retrieval-Augmented Generation (RAG) pipeline to help users find recipes. Developed as a test-task. 

## Solution Summarization
Techniques and conclusions that led to the presented solution are thoroughly described in [Summary Notebook](https://github.com/anyaachan/recipe-chatbot/blob/main/notebooks/summary.ipynb).

## Running the Chatbot
To start a chatbot in CLI:

1. Clone the repository.
2. Create a virtual environment and install dependencies from ```requirements.txt```.
3. Put your data in `data/input/Recipes.csv`
3. Create a `.env` file with your API keys:
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```
4. Run main.py
