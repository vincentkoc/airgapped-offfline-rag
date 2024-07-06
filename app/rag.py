from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from fastembed import TextEmbedding
from utils import load_config

config = load_config()

def get_embedding_function():
    if config['use_fast_embed']:
        return TextEmbedding()
    else:
        return HuggingFaceEmbeddings(model_name=config['embedding_model'])

def retrieve_context(query, top_k=3):
    embeddings = get_embedding_function()
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    docs = vectorstore.similarity_search(query, k=top_k)
    context = "\n".join([doc.page_content for doc in docs])

    return context
