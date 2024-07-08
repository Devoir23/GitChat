import os
import gc
import re
import uuid
import subprocess
import requests
from dotenv import load_dotenv

os.environ["HF_HOME"] = "weights"
os.environ["TORCH_HOME"] = "weights"
import streamlit as st

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext

from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding

from rag_101.retriever import (
    load_embedding_model,
    load_reranker_model
)

# setup the llm
ollama_url = 'http://localhost:11434/api/chat'

llm = Ollama(model="mistral:instruct", url=ollama_url ,request_timeout=1000.0)

# TODO: setup the embedding model
lc_embedding_model = load_embedding_model()
embed_model = LangchainEmbedding(lc_embedding_model)


# utility functions
def parse_github_url(url):
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, url)
    return match.groups() if match else (None, None)


def clone_repo(repo_url):
    try:
        result = subprocess.run(["git", "clone", repo_url], check=True, text=True, capture_output=True)
        print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")
        raise e


def validate_owner_repo(owner, repo):
    return bool(owner) and bool(repo)


if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None


def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


with st.sidebar:

    # input for Github URL
    github_url = st.text_input("Github Repository URL")

    # button to load and process the github repository
    process_button = st.button("Load")

    message_container = st.empty()  # placeholder for dynamic messages

    if process_button and github_url:
        owner, repo = parse_github_url(github_url)

        if validate_owner_repo(owner, repo):
            with st.spinner(f"Loading {repo} repository by {owner}..."):
                try:
                    input_dir_path = f"./{repo}"

                    if not os.path.exists((input_dir_path)):
                        clone_repo(github_url)

                    if os.path.exists(input_dir_path):
                        loader = SimpleDirectoryReader(
                            input_dir=input_dir_path,
                            required_exts=[".py", ".ipynb", ".js", ".ts", ".md"],
                            recursive=True
                        )
                    else:
                        st.error('Error occurred while cloning the repo, carefully check the URL')
                        st.stop()

                    docs = loader.load_data()

                    # TODO: ====== Create vector store and upload data ======
                    Settings.embed_model = embed_model
                    index = VectorStoreIndex.from_documents(docs)

                    # setup a query engine
                    Settings.llm = llm
                    query_engine = index.as_query_engine(streaming=True, similarity_top_k=4)

                    # customize prompt template
                    qa_prompt_tmpl_str = (
                        "Context information is below.\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "Given the context information above I want you to think step by step to answer the query in a crisp manner, in case you don't know the answer say 'I don't know!'.\n"
                        "Query: {query_str}\n"
                        "Answer: "
                    )
                    qa_prompt_tmpl_str = PromptTemplate(qa_prompt_tmpl_str)

                    query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": qa_prompt_tmpl_str}
                    )

                    if docs:
                        message_container.success("Data loaded successfully!!")
                    else:
                        message_container.write(
                            "No Data found, check if repository is not empty!!"
                        )
                    st.session_state.query_engine = query_engine

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.stop()

                st.success("Ready to chat!")
        else:
            st.error('Invalid owner or repo')
            st.stop()

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"GitChat🌐")
    st.header(f"Chat with your code! </>")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
# TODO: old one
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # context = st.session_state.context
        query_engine = st.session_state.query_engine

        # Simulate stream of response with milliseconds delay
        streaming_response = query_engine.query(prompt)

        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        # full_response = query_engine.query(prompt)

        message_placeholder.markdown(full_response)
        # st.session_state.context = ctx

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# todo: new one
# prompt = st.chat_input("What's up?")
# if prompt:
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
#
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(prompt)
#
#     # Display assistant response in chat message container
#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         full_response = ""
#
#     # context
#     query_engine = st.session_state.query_engine
#
#     # simulate stream of response with milliseconds delay
#     try:
#         # Construct the request payload
#         payload = {
#             "message": prompt,
#             "model": "mistral:instruct"
#         }
#
#         # Send the request
#         response = requests.post(ollama_url, json=payload)
#
#         # Check for HTTP errors
#         response.raise_for_status()
#
#         # Print the full response to debug
#         response_json = response.json()
#         print(response_json)
#
#         # Process the response
#         if "response_gen" in response_json:
#             for chunk in response_json["response_gen"]:
#                 full_response += chunk
#                 message_placeholder.markdown(full_response + "▌")
#             message_placeholder.markdown(full_response)
#
#             # add assistant response to chat history
#             st.session_state.messages.append({"role": "assistant", "content": full_response})
#         else:
#             st.error("Unexpected response format: 'response_gen' key not found")
#
#     except requests.exceptions.HTTPError as e:
#         st.error(f"HTTP error: {e}")


