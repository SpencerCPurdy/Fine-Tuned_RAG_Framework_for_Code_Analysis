Live Demo: https://huggingface.co/spaces/SpencerCPurdy/Fine-Tuned_RAG_Framework_for_Code_Analysis

# Fine-Tuned RAG Framework for Code Analysis

This project is a production-ready, Retrieval-Augmented Generation (RAG) system specifically designed for code analysis and software development queries. It integrates a fine-tuned Large Language Model (LLM) with a vector database to provide accurate, context-aware answers based on a comprehensive knowledge base of programming concepts.

A key feature of this system is its **automatic fine-tuning process**. On initialization, it automatically fine-tunes the base model (`Salesforce/codegen-350M-mono`) on a curated dataset of code-related questions and answers using Parameter-Efficient Fine-Tuning (PEFT) with LoRA. This ensures the model is specialized for the domain of software engineering, resulting in higher quality and more relevant responses.

## Core Features

* **Automatic Model Fine-Tuning**: The system automatically fine-tunes a code-specific language model (`Salesforce/codegen-350M-mono`) on startup using LoRA to specialize it for code analysis tasks.
* **Retrieval-Augmented Generation (RAG)**: Leverages a `ChromaDB` vector store and a `sentence-transformers` model to retrieve the most relevant documents from a knowledge base to ground the LLM's responses in factual information.
* **Code-Specific Knowledge Base**: The system is pre-loaded with a detailed knowledge base covering software architecture patterns, clean code principles, testing strategies, performance optimization, and API design best practices.
* **Comprehensive Evaluation Metrics**: Every generated response is evaluated in real-time for relevance, context grounding, hallucination, and technical accuracy. The system also calculates a performance improvement score based on whether the fine-tuned model is active.
* **Performance & Cost Tracking**: A built-in `PerformanceTracker` monitors system usage, including query latency, tokens processed, and estimated operational costs, providing a full overview of system efficiency.
* **Source Attribution**: To ensure transparency and trustworthiness, every answer is accompanied by a list of the source documents from the knowledge base that were used to generate the response.

## How It Works

The system follows an automated pipeline from startup to query processing:

1.  **Automatic Fine-Tuning (On Startup)**: The `ModelFineTuner` class initiates an automatic fine-tuning process. It loads the base `Salesforce/codegen-350M-mono` model, applies a LoRA configuration, and trains it on a specialized dataset of code analysis Q&A pairs. The resulting fine-tuned model is then used for generation.
2.  **Knowledge Base Indexing**: The `RAGSystem` class initializes a `ChromaDB` vector store. It processes and chunks the provided code documentation, computes embeddings using a `SentenceTransformer` model, and indexes these embeddings for efficient retrieval.
3.  **Query Processing**: A user submits a query through the Gradio interface.
4.  **Retrieval**: The system encodes the user's query into an embedding and uses it to search the `ChromaDB` vector store, retrieving the top-k most relevant text chunks from the knowledge base.
5.  **Generation**: The retrieved chunks are formatted as context and prepended to the user's query in a prompt. This combined prompt is then passed to the fine-tuned LLM, which generates a context-aware and accurate answer.
6.  **Evaluation & Display**: The final response is evaluated for quality, and all relevant information—the answer, sources, metrics, and performance data—is presented to the user in the interactive dashboard.

## Technical Stack

* **AI & Machine Learning**: `transformers`, `peft`, `bitsandbytes`, `torch`, `accelerate`
* **Retrieval & Vector Search**: `chromadb`, `sentence-transformers`, `langchain`
* **Data Processing**: `pandas`, `numpy`, `datasets`
* **Web Interface & Dashboard**: `gradio`

## How to Use the Demo

The interface is designed for simplicity and provides a wealth of information with each query.

1.  Enter a question related to software development, architecture, or best practices into the text box. You can use the provided sample queries as inspiration.
2.  Click the **Analyze Query** button.
3.  Review the output in the panels:
    * **Analysis Result**: The generated answer from the fine-tuned RAG system.
    * **Referenced Sources**: The documents from the knowledge base used to formulate the answer.
    * **Response Metrics**: A detailed breakdown of the response quality, including relevance, grounding, and technical accuracy scores.
    * **Performance Data**: Information on the query processing time, tokens used, and estimated cost.
    * **System Statistics**: An overview of cumulative usage and performance.

## Disclaimer

This project is a demonstration of a production-ready RAG system for a specialized domain. The automatic fine-tuning process is computationally intensive and may be slow on CPU. For optimal performance, running this demo on a GPU-accelerated environment is recommended. All generated responses are for informational purposes and should be validated by a qualified professional.
