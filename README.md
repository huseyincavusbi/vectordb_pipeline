# vectordb_pipeline

This repository contains a Jupyter Notebook that builds a persistent vector database for a medical Retrieval-Augmented Generation (RAG) system. It processes and embeds two key medical datasets using GPU acceleration in a Kaggle environment and stores them in separate ChromaDB collections.

## Key Features
-   **⚕️ Dual Medical Datasets**: Ingests clinical guidelines and medical textbook excerpts to create a comprehensive knowledge base.
    -   `epfl-llm/guidelines`: A dataset of clinical practice guidelines.
    -   `MedRAG/textbooks`: A dataset containing content from various medical textbooks.
-   **⚡ GPU-Optimized Embedding**: Leverages `sentence-transformers/all-MiniLM-L6-v2` with CUDA for rapid and efficient document embedding.
-   **🗄️ Persistent Vector Storage**: Creates and saves two distinct, persistent [ChromaDB](https://www.trychroma.com/) collections for organized and efficient retrieval.
-   **📦 Ready for Deployment**: The final output is a compressed `chroma_db.zip` archive, making it easy to download and integrate into any RAG application.

## What the Notebook Does
1.  **Environment Setup**: Installs all necessary Python libraries, including `langchain`, `chromadb`, and `sentence-transformers`.
2.  **Data Loading & Processing**: Loads the two medical datasets from the Hugging Face Hub.
3.  **Text Chunking**: Splits the medical documents into smaller, manageable chunks optimized for embedding.
4.  **Embedding Generation**: Uses a pre-trained sentence transformer model to convert the text chunks into vector embeddings.
5.  **Vector Store Creation**: Ingests the embeddings into two separate ChromaDB collections:
    -   `medical_guidelines`
    -   `medical_textbooks`
6.  **Verification & Packaging**: Confirms the successful creation and population of the ChromaDB collections and packages the entire database into a single `.zip` file for easy portability.

## How to Use
### Running the Notebook
1.  Open the `chromadb-data-ingestion.ipynb` notebook in a Kaggle environment with a GPU accelerator (e.g., T4).
2.  Run all the cells in the notebook sequentially.
3.  Once the execution is complete, a `chroma_db.zip` file will be generated in the `/kaggle/working/` directory.

### Using the Output Database
1.  Download the `chroma_db.zip` file from your Kaggle notebook's output.
2.  Unzip the file in your project directory.
3.  You can then load the persistent ChromaDB collections in your own application like this:

```python
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize the same embedding model used in the notebook
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda'}, # or 'cpu'
    encode_kwargs={'normalize_embeddings': True}
)

# Load the persisted vector stores
guidelines_db = Chroma(
    persist_directory="./chroma_db",
    collection_name="medical_guidelines",
    embedding_function=embedding_model
)

textbooks_db = Chroma(
    persist_directory="./chroma_db",
    collection_name="medical_textbooks",
    embedding_function=embedding_model
)

# You can now perform similarity searches on the loaded databases
retrieved_docs = guidelines_db.similarity_search("treatment for hypertension")
```

## Customization and Adaptation

This pipeline is designed to be flexible and can be easily adapted for different use cases:

### Using Different Embedding Models
You can replace the default `sentence-transformers/all-MiniLM-L6-v2` model with other embedding models. Simply modify the model initialization in the notebook:

```python
# Example: Using a different sentence-transformers model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",  # Higher quality, slower
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

# Example: Using OpenAI embeddings (requires API key)
from langchain_openai import OpenAIEmbeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Example: Using other Hugging Face models
embedding_model = HuggingFaceEmbeddings(
    model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",  # Medical domain-specific
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)
```

### Using Different Datasets
The pipeline can be adapted to work with any text dataset. To use different datasets:

1. **Replace the dataset loading code** with your own data source:
```python
# Example: Loading from local files
import pandas as pd
df = pd.read_csv("your_dataset.csv")
documents = df['text_column'].tolist()

# Example: Loading from other Hugging Face datasets
from datasets import load_dataset
dataset = load_dataset("your-username/your-dataset")
documents = dataset['train']['text']

# Example: Loading from JSON files
import json
with open("your_data.json", "r") as f:
    data = json.load(f)
    documents = [item['content'] for item in data]
```

2. **Adjust the collection names** to reflect your new datasets:
```python
collection_name = "your_custom_collection"
```

3. **Modify text preprocessing** if needed based on your data format and requirements.

### Supported Data Formats
The pipeline can handle various data formats including:
- Hugging Face datasets
- CSV files
- JSON files
- Plain text files
- PDF documents (with additional processing)
- Web scraping results

## Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended for faster processing)
- At least 8GB of available storage space
- Internet connection for downloading models and datasets

## License

MIT License
