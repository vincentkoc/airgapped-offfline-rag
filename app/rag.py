from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from utils import load_config

config = load_config()

def retrieve_context(query, top_k=3):
    embeddings = HuggingFaceEmbeddings(model_name=config['embedding_model'])
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    docs = vectorstore.similarity_search(query, k=top_k)
    context = "\n".join([doc.page_content for doc in docs])

    return context
