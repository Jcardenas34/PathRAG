# app.py
# Import necessary libraries
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
# import argparse
# from langchain_postgres import PGVector
# from pgvector.psycopg import register_vector
# from langchain_openai import OpenAIEmbeddings
# from langchain.chat_models import init_chat_model

# import bs4
# from langchain import hub
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langgraph.graph import START, StateGraph
# from typing_extensions import List, TypedDict


import os
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import gpt_4o_mini_complete



llm_model="gpt-4.1"

# # Not importing from local package
# class RetrieveApiKey:
#     """Class to retrieve API keys for different models."""

#     def __init__(self, model:str):
#         self.model = model

#         self.supported_models = {"gpt-4.1":  os.environ.get("OPENAI_API_KEY_PROPETERRA"),
#                                  "o4-mini":  os.environ.get("OPENAI_API_KEY_PROPETERRA"),
#                                  "sonar":    os.environ.get("PERPLEXITY_API_KEY_PROPETERRA"),
#                                  "sonar-pro":os.environ.get("PERPLEXITY_API_KEY_PROPETERRA"),
#                                  "sonar-deep-research":os.environ.get("PERPLEXITY_API_KEY_PROPETERRA"),
#                                  "ms_copilot":os.environ.get("MS_COPILOT_API_KEY"),
#                                  "mistral":         os.environ.get("MISTRAL_API_KEY"),
#                                  "gemini-2.5-flash":os.environ.get("GEMINI_API_KEY"),
#                                  "gemini-2.5-pro":os.environ.get("GEMINI_API_KEY"),
#                                  "manus":           os.environ.get("MANUS_API_KEY")}

#         if model not in self.supported_models:

#             raise ValueError("Selected model not in list of supported models.")


# # Define state for application (Taken from https://python.langchain.com/docs/tutorials/rag/)
# class State(TypedDict):
#     question: str
#     context: List[Document]
#     answer: str




# def initialize(model:str):
#     ''' Initializes the model used for querying, as well as the vector database used for storage. '''


#     # Initializing necessary parameters, at the global level
#     key_retriever = RetrieveApiKey(model)
#     # For now, only gpt and gemini models work, given that you have your api keys set in your environment for them
#     model_key = key_retriever.supported_models[model]

#     # Hardcoded, for gpt, must use this if you want to embed with openai
#     openai_key = key_retriever.supported_models["gpt-4.1"]

#     if "gpt" in model or "o4-mini" in model:
#         llm = init_chat_model(model, model_provider="openai", api_key=model_key)
#     elif "gemini" in model:
#         llm = init_chat_model(model, model_provider="google_genai", api_key=model_key)
#     else:
#         print("Please choose a supported model")

#     embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_key)

#     # Connecting to a postgres database you have defined (must also add teh vector extension! run 'CREATE EXTENSION vector;' once db is open)
#     # https://www.postgresql.org/docs/current/tutorial-install.html
#     connection = "postgresql+psycopg://chiral@localhost:6543/postgres"
#     collection_name = "my_docs"

#     vector_store = PGVector(
#         embeddings=embeddings,
#         collection_name=collection_name,
#         connection=connection,
#     )


#     # (Taken from https://python.langchain.com/docs/tutorials/rag/)
#     # Define prompt for question-answering
#     # N.B. for non-US LangSmith endpoints, you may need to specify
#     # api_url="https://api.smith.langchain.com" in hub.pull.
#     prompt = hub.pull("rlm/rag-prompt")

#     return prompt, llm, vector_store

# def index_data(vector_store, site):
#     # Scrapes the content of the provided link, must change  class_=("ArticleContent") to relevant sections you want to scrape
#     print(f"Indexing site: {site}")
#     loader = WebBaseLoader(
#         # web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#         web_paths=((site,)),
#         bs_kwargs=dict(
#             parse_only=bs4.SoupStrainer(
#                 class_=("ArticleContent")
#                 # class_=("post-title", "post-header", "post-content")
#             )
#         ),
#     )

#     docs = loader.load()
#     assert len(docs) == 1

#     # What are the contents of the graph?
#     # if args.verbose:
#     #     print(f"Total characters: {len(docs[0].page_content)}")
#     #     print(docs[0].page_content[:1000])


#     # Split text into chunks (tokens) so that they can be vectorized 
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     all_splits = text_splitter.split_documents(docs)

#     # Start indexing the chunks
#     document_ids = vector_store.add_documents(documents=all_splits)
#     # if args.verbose:
#         # print(document_ids[:3])



# # --- Your RAG Logic Goes Here ---
# # This is a placeholder function.
# # Replace this with your actual RAG script's logic.
# def get_rag_response(question: str, llm_model:str) -> str:
#     """
#     This function takes a user prompt, processes it through the RAG system,
#     and returns the generated response.

#     Args:
#         prompt: The input string from the user.

#     Returns:
#         The RAG model's response as a string.
#     """

#     indexed_sites = ["","https://thelatinvestor.com/blogs/news/colombia-real-estate-market",
#                      "https://thelatinvestor.com/blogs/news/argentina-real-estate-foreigner",
#                      "https://thelatinvestor.com/blogs/news/panama-city-property"]
    
#     # model = "gemini-2.5-pro"
    
    
#     prompt, llm, vector_store = initialize(llm_model)

#     # Define application steps (Taken from https://python.langchain.com/docs/tutorials/rag/)
#     def retrieve(state: State):
#         retrieved_docs = vector_store.similarity_search(query=state["question"], k=5, ) # filter=None will filter by metadata stored, use for only retrieving specific countries, add to meta data
#         return {"context": retrieved_docs}


#     # (Taken from https://python.langchain.com/docs/tutorials/rag/)
#     def generate(state: State):
#         docs_content = "\n\n".join(doc.page_content for doc in state["context"])
#         messages = prompt.invoke({"question": state["question"], "context": docs_content})
#         response = llm.invoke(messages)
#         return {"answer": response.content}


#     # site = "https://thelatinvestor.com/blogs/news/colombia-real-estate-market"
#     # site = "https://thelatinvestor.com/blogs/news/argentina-real-estate-foreigner"
#     site = "https://thelatinvestor.com/blogs/news/panama-city-property"

#     if site not in indexed_sites:
#         print(f"Site {site} not found in database, will load it and parse for an answer to your question.")
#         index_data(vector_store, site)
#     else:
#         print(f"Site {site} found in vector store, loading it..")




#     # Compile application and test (Taken from https://python.langchain.com/docs/tutorials/rag/)
#     graph_builder = StateGraph(State).add_sequence([retrieve, generate])
#     graph_builder.add_edge(START, "retrieve")
#     graph = graph_builder.compile()

#     # Getting the actual response from the model
#     response = graph.invoke({"question": question})
#     print(response["answer"])



#     print(f"Received prompt: {prompt}")
#     # For now, we'll just echo the prompt back with a prefix.
#     # TODO: Replace this with your actual RAG implementation.
#     # For example: response = my_rag_chain.invoke(prompt)
#     # response = f"This is the RAG system's answer to: '{prompt}'"
#     # print(f"Generated response: {response}")
#     return response["answer"]
# # ---------------------------------


# Initialize the Flask application
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS)
# This is crucial to allow your React frontend (running on a different port)
# to communicate with this Flask backend.
CORS(app)
# --- API Endpoints to Control the LLM ---

@app.route('/api/llm', methods=['GET'])
def get_llm():
    """
    Endpoint to get the currently selected LLM.
    """
    global llm_model
    return jsonify({"llm": llm_model})

@app.route('/api/llm', methods=['POST'])
def set_llm():
    """
    Endpoint to update the selected LLM.
    It expects a JSON payload like: {"llm": "gpt-4.1"}
    """
    global llm_model
    data = request.get_json()
    
    new_llm = data.get('llm')
    if not new_llm:
        return jsonify({"error": "LLM name is missing"}), 400
        
    llm_model = new_llm
    print(f"Selected LLM changed to: {llm_model}")
    return jsonify({"message": f"LLM successfully changed to {llm_model}"})

# ------------------------------------------------

# Define the API endpoint for the chat functionality
@app.route('/api/chat', methods=['POST'])
def chat():
    """
    This endpoint receives a prompt from the frontend,
    gets a response from the RAG system, and sends it back.
    """
    try:

        global llm_model
        # Get the JSON data from the request
        data = request.get_json()
        
        # Extract the prompt from the data.
        # It's good practice to validate the input.
        prompt = data.get('prompt')
        if not prompt:
            return jsonify({"error": "Prompt is missing"}), 400

        rag = PathRAG(
            working_dir=WORKING_DIR,
            llm_model_func=gpt_4o_mini_complete,  
        )
        # Get the response from your RAG function
        # response_text = get_rag_response(prompt, llm_model)
        response_text = rag.query(prompt, param=QueryParam(mode="hybrid"))

        # Return the response as JSON
        return jsonify({"response": response_text})

    except Exception as e:
        # Log the error for debugging
        print(f"An error occurred: {e}")
        # Return a generic error message
        return jsonify({"error": "An internal server error occurred"}), 500

# This block ensures the server runs only when the script is executed directly
if __name__ == '__main__':

    WORKING_DIR = "/Users/chiral/git_projects/PathRAG"

    # api_key=""
    # os.environ["OPENAI_API_KEY"] = api_key
    base_url="https://api.openai.com/v1"
    os.environ["OPENAI_API_BASE"]=base_url


    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    # Run the Flask app on port 5000 in debug mode for development
    # Host '0.0.0.0' makes it accessible on your local network
    app.run(host='0.0.0.0', port=5000, debug=True)
