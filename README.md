# Chat with Your Code - RAG Application

Transform your interaction with GitHub repositories through a natural language interface. This "Chat with your code" Retrieval-Augmented Generation (RAG) application simplifies code queries, making coding more intuitive and productive.

![Demo Video](//link_to_demo_video)

## Key Architecture Components

Building a robust RAG application involves several key components. The architecture diagram below illustrates how these components interact, followed by detailed descriptions of each component.

![Architecture Diagram](//link_to_architecture_diagram)

### 1. Custom Knowledge Base

The Custom Knowledge Base is a collection of relevant and up-to-date information that serves as a foundation for the RAG system. In this case, the knowledge base is a GitHub repository used as a source of truth to provide answers to user queries.

### 2. Chunking

Chunking is the process of breaking down a large input text into smaller pieces. This ensures that the text fits the input size of the embedding model and improves retrieval efficiency.

### 3. Embeddings Model

The Embeddings Model represents text data as numerical vectors, which can be input into machine learning models. The embedding model is responsible for converting text into these vectors.

### 4. Vector Databases

A Vector Database is a collection of pre-computed vector representations of text data for fast retrieval and similarity search. It includes capabilities like CRUD operations, metadata filtering, and horizontal scaling. By default, LlamaIndex uses a simple in-memory vector store that is great for quick experimentation.

### 5. User Chat Interface

The User Chat Interface is a user-friendly interface that allows users to interact with the RAG system, providing input queries and receiving outputs. We have built a Streamlit app to facilitate this interaction. The code for the app can be found in `app.py`.

### 6. Query Engine

The Query Engine takes the query string, fetches relevant context, and sends both as a prompt to the Large Language Model (LLM) to generate a final natural language response. The LLM used here is Mistral-7B, which is served locally using Ollama. The final response is displayed in the user interface.

### 7. Prompt Template

A custom prompt template is used to refine the response from the LLM and include the necessary context.

## Conclusion

we developed a Retrieval-Augmented Generation (RAG) application that allows you to "Chat with your code." Throughout this process, we explored LlamaIndex, the go-to library for building RAG applications, and Ollama for locally serving LLMs.

We also delved into the concept of prompt engineering to refine and steer the responses of our LLM. These techniques can similarly be applied to anchor your LLM to various knowledge bases, such as documents, PDFs, videos, and more.

LlamaIndex has a variety of data loaders. You can learn more about them [here](https://link_to_llamaindex_data_loaders).

## Getting Started

### Prerequisites

- Python 3.8 or above
- Git
- Streamlit
- LlamaIndex
- Ollama
- Mistral-7B model

### Installation
1. Clone the repository:
```bash 
git clone https://github.com/yDevoir23/GitChat.git
cd GitChat
```
2. Install the required packages:
```bash
streamlit run app.py
```
## Running the Application
1. Start the Streamlit app:
```bash
streamlit run app.py
```
2. Follow the on-screen instructions to interact with the application.

## Conclusion

In this studio, we developed a Retrieval-Augmented Generation (RAG) application that allows you to "Chat with your code." Throughout this process, we explored LlamaIndex, the go-to library for building RAG applications, and Ollama for locally serving LLMs.

We also delved into the concept of prompt engineering to refine and steer the responses of our LLM. These techniques can similarly be applied to anchor your LLM to various knowledge bases, such as documents, PDFs, videos, and more.

LlamaIndex has a variety of data loaders. You can learn more about them [here](https://link_to_llamaindex_data_loaders).
   
## Acknowledgments
LlamaIndex
Ollama
Mistral-7B

