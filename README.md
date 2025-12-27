
# University Assistant Chatbot

This is a custom chatbot for university students that provides accurate, precise, and comprehensive information about university procedures, fee structures, and other university-related information.

I used the RAG (Retreival Augemented Generation) pipeline for this project and it uses Google Cloud API for the responses.

Currently, it is on Streamlit and I am planning to shift it to my own custom web design.

## Features

- **Enhanced Accuracy**: Uses advanced models and optimized processing for highly accurate responses
- **Improved Embeddings**: Uses BAAI/bge-base-en-v1.5 model for better semantic understanding
- **Relevance Filtering**: Automatically filters out irrelevant chunks to improve answer quality
- **Multi-Source Knowledge Base**: Combines data from both PDF documents and university website
- **Semantic Search**: Uses FAISS for efficient similarity search of document chunks
- **Source Filtering**: Option to filter responses based on source type (PDF, web, or both)
- **User-Friendly Interface**: Clean Streamlit interface for easy interaction
- **Beginner-Friendly Responses**: Responses are formatted to be easy to understand
- **Organized Output**: Information is presented in a structured way
- **Debug Mode**: Advanced troubleshooting for fine-tuning relevance thresholds and viewing chunk selection

## Requirements

- Python 3.12.0
- Google Gemini API key (you can get one from [Google AI Studio](https://makersuite.google.com/app/apikey))

## Setup and Installation

1. Clone this repository or download the files
2. Create a virtual environment by using this command below:
   ```
   python -m venv .venv
   ```
3. Activate the virtual environment:
   ```
   .\\.venv\\Scripts\\activate.bat
   ```
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Set up your Google Gemini API key in one of these ways:
   - Create a `.env` file with `GOOGLE_API_KEY=your_api_key_here`
   - Enter the API key in the `.env` file

4. Create a `Data` folder in the parent directory and place the `Links.txt` file for website data
   - Create a folder `PDF's` inside `Data` folder and place all your PDF's

## API Key Configuration

This application uses the Google Gemini API key stored in a `.env` file. Follow these steps to set up your API key:

1. Copy the `env.example` file to a new file named `.env`:
   ```
   cp env.example .env
   ```

2. Open the `.env` file in a text editor.

3. Replace the example API key with your actual Google Gemini API key:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

4. Save the file. The application will automatically use this API key for all users.

## Usage

### Step 1: Process Data and Create Embeddings

First, you need to process the PDF and web data, then create the embeddings database:

```
python process_pdfs.py
```

This will:
- Extract text from all PDFs in the `Data/PDF's` and `Data/Fee_structure` directories
- Scrape content from all URLs listed in `Data/Links.txt`
- Split the text into optimized chunks (800 characters with 250 character overlap)
- Generate embeddings using the all-MiniLM-L6-v2 model
- Save the FAISS index and chunks for later use

### Step 2: Start the Chatbot Interface

```
streamlit run app.py
```

This will open a web interface where you can:
1. Enter your Google Gemini API key (if not provided via .env file)
2. Load the knowledge base
3. Choose which data sources to use (PDFs, website, or both)
4. Ask questions about your university

## How It Works

1. **Data Processing**:
   - PDF Processing: PDFs are read and text is extracted
   - Web Scraping: Content is scraped from university website URLs
   - Text Chunking: Documents are split into smaller optimized chunks with context-preserving overlap

2. **Embedding and Search**:
   - High-Quality Embeddings: Chunks are converted to vector embeddings using BAAI/bge-base-en-v1.5
   - Enhanced Similarity Search: When a question is asked, the system finds semantically relevant chunks
   - Relevance Filtering: Chunks below the relevance threshold are filtered out to prevent hallucinations

3. **Response Generation**:
   - Organized Context: The system provides context to the model organized by source type and relevance
   - Strict Guardrails: The system enforces strict guidelines to prevent making up information not in the context
   - Comprehensive Response: The Gemini 1.5 Pro model generates accurate, beginner-friendly responses

4. **User Interface**:
   - Streamlit provides a clean, user-friendly chat interface
   - Source filtering allows focusing on specific types of information

## Project Structure

- `Backend\`: Directory containing all backend processing files
   - `embeddings_manager.py`: Module for managing embeddings and FAISS index
   - `gemini_api.py`: Module for interacting with Google's Gemini API
   - `process_pdfs.py`: Script to process PDFs and web content, then generate embeddings
   - `web_scraper.py`: Module for scraping and processing web content
- `app.py`: Main Streamlit application
- `pdf_loader.py`: Module for loading and processing PDFs
- `Data/`: Directory containing university documents and links
   - `PDF's/`: General university documents
   - `Fee_structure/`: Fee structure documents for different departments
   - `Links.txt`: List of university website URLs to scrape

## Customization

- Adjust chunk size and overlap in `process_pdfs.py` for different document types
- Modify the prompt template in `gemini_api.py` to change the response style
- Add more URLs to `Data/Links.txt` to expand the knowledge base
- Update the CSS in `app.py` to change the appearance of the chatbot

## Troubleshooting

If you encounter any issues:

- Ensure your Google Gemini API key is valid
- Check that the PDFs and web content have been processed by running `process_pdfs.py`
- Verify that the `embeddings` directory contains the FAISS index files
- Make sure all required packages are installed 