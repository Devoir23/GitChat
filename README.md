# GitChat - RAG Application

Transform your interaction with GitHub repositories through a natural language interface. This "Chat with your code" Retrieval-Augmented Generation (RAG) application simplifies code queries, making coding more intuitive and productive.

# Demo Video

<a href="https://www.youtube.com/watch?v=5A47-m0MWiM" target="_blank">
  <img src="https://img.youtube.com/vi/5A47-m0MWiM/0.jpg" alt="Demo Video" style="width:866px;height:480px;">
</a>


## Key Architecture Components

Building a robust RAG application involves several key components. The architecture diagram below illustrates how these components interact, followed by detailed descriptions of each component.

![image](https://github.com/Devoir23/GitChat/assets/83571014/f0df37da-b5a7-4487-bb04-f0e60ea4ab61)


### 1. Custom Knowledge Base

The Custom Knowledge Base is a collection of relevant and up-to-date information that serves as a foundation for the RAG system. In this case, the knowledge base is a GitHub repository used as a source of truth to provide answers to user queries.

### 2. Chunking

Chunking is the process of breaking down a large input text into smaller pieces. This ensures that the text fits the input size of the embedding model and improves retrieval efficiency.

### 3. Embeddings Model

The Embeddings Model represents text data as numerical vectors, which can be input into machine learning models. The embedding model is responsible for converting text into these vectors.
```bash
 model_name: str = "BAAI/bge-large-en-v1.5", device: str = "cuda"
) -> HuggingFaceBgeEmbeddings:
    model_kwargs = {"device": device}
    encode_kwargs = {
        "normalize_embeddings": True
    }  # set True to compute cosine similarity
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
```

### 4. Vector Databases

A Vector Database is a collection of pre-computed vector representations of text data for fast retrieval and similarity search. It includes capabilities like CRUD operations, metadata filtering, and horizontal scaling. By default, LlamaIndex uses a simple in-memory vector store that is great for quick experimentation.
```bash
# ====== Create vector store and upload indexed data ======
Settings.embed_model = embed_model # we specify the embedding model to be used
index = VectorStoreIndex.from_documents(docs)
```

### 5. User Chat Interface

The User Chat Interface is a user-friendly interface that allows users to interact with the RAG system, providing input queries and receiving outputs. We have built a Streamlit app to facilitate this interaction. The code for the app can be found in `app.py`.

### 6. Query Engine

The Query Engine takes the query string, fetches relevant context, and sends both as a prompt to the Large Language Model (LLM) to generate a final natural language response. The LLM used here is Mistral-7B, which is served locally using Ollama. The final response is displayed in the user interface.
```bash
# setting up the llm
llm = Ollama(model="mistral", request_timeout=1000.0) 

# ====== Setup a query engine on the index previously created ======
Settings.llm = llm # specifying the llm to be used
query_engine = index.as_query_engine(streaming=True, similarity_top_k=4)
```

### 7. Prompt Template

A custom prompt template is used to refine the response from the LLM and include the necessary context.
```bash
qa_prompt_tmpl_str = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
            "Query: {query_str}\n"
            "Answer: "
            )

qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

response = query_engine.query('What is this repository about?')
print(response)
```
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
pip install -r requirements.txt
```
## Running the Application
1. Start the Streamlit app:
```bash
streamlit run app.py
```
2. Follow the on-screen instructions to interact with the application.

## Conclusion

we developed a Retrieval-Augmented Generation (RAG) application that allows you to "**Chat with your code.**" Throughout this process, we explored LlamaIndex, the go-to library for building RAG applications, and Ollama for locally serving LLMs.

**Here we are running this application on RTX 3050 GPU. It may vary in your device based on specifications of your device.**

We also delved into the concept of prompt engineering to refine and steer the responses of our LLM. These techniques can similarly be applied to anchor your LLM to various knowledge bases, such as documents, PDFs, videos, and more.

**LlamaIndex** has a variety of data loaders. You can learn more about them [here.](https://docs.llamaindex.ai/en/stable/understanding/loading/loading/)
   
## Acknowledgments
- LlamaIndex
- Ollama
- Mistral-7B

## Connect with me
<p align="left">
<a href="https://twitter.com/kartavyarb" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/twitter.svg" alt="kartavyarb" height="30" width="40" /></a>
<a href="https://instagram.com/kartavya.here" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/instagram.svg" alt="kartavya.here" height="30" width="40" /></a>
</p>
