# Basic RAG Pipeline with Google Gemini

This project implements a basic Retrieval-Augmented Generation (RAG) pipeline using LangChain and Google Gemini models.

## Setup

1. **Clone the repository (if applicable) or download the files.**
2. **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up your Google API Key:**
    Create a file named `.env` in the `rag_pipeline` directory and add your Google API key (usually an API key for Google AI Studio or Vertex AI):

    ```bash
    echo "GOOGLE_API_KEY='your_google_api_key_here'" > .env
    ```

    You can obtain a Google API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Usage

Run the main script:

```bash
python main.py
```

The script will load documents from the `docs` directory, process them using Google Gemini for embeddings and generation, and then you can ask questions based on their content.
