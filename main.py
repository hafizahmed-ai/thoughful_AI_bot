from fastapi import FastAPI, Header, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
import logging
import traceback
from utils import (Qdrant_DB, EmbeddingModel, gpt_chain)
import json
from contextlib import asynccontextmanager
from langchain.docstore.document import Document

embedding_model = EmbeddingModel()

VECTOR_STORES = {}

collection_id="new_json_collection"
# The JSON data (either hardcoded or read from a .json file)
json_data = {
    "questions": [
        {
            "question": "What does the eligibility verification agent (EVA) do?",
            "answer": "EVA automates the process of verifying a patient’s eligibility and benefits information in real-time, eliminating manual data entry errors and reducing claim rejections."
        },
        {
            "question": "What does the claims processing agent (CAM) do?",
            "answer": "CAM streamlines the submission and management of claims, improving accuracy, reducing manual intervention, and accelerating reimbursements."
        },
        {
            "question": "How does the payment posting agent (PHIL) work?",
            "answer": "PHIL automates the posting of payments to patient accounts, ensuring fast, accurate reconciliation of payments and reducing administrative burden."
        },
        {
            "question": "Tell me about Thoughtful AI's Agents.",
            "answer": "Thoughtful AI provides a suite of AI-powered automation agents designed to streamline healthcare processes. These include Eligibility Verification (EVA), Claims Processing (CAM), and Payment Posting (PHIL), among others."
        },
        {
            "question": "What are the benefits of using Thoughtful AI's agents?",
            "answer": "Using Thoughtful AI's Agents can significantly reduce administrative costs, improve operational efficiency, and reduce errors in critical processes like claims management and payment posting."
        }
    ]
}

def generate_documents_from_json(json_data) -> list:
    """
    Generate and structure documents from JSON data.

    Parameters:
    - json_data (dict): The JSON data containing questions and answers.

    Returns:
    - List[Document]: List of documents where each document represents a Q/A pair.
    """
    logging.info("Generating documents from JSON data")
    try:
        documents = []

        # Iterate over each Q/A pair in the JSON data
        for entry in json_data.get('questions', []):
            question = entry.get('question')
            answer = entry.get('answer')

            doc = Document(
                page_content=question+' '+answer,  # This will be used for similarity search
                metadata={"answer": answer}  # Store the answer in metadata
            )

            documents.append(doc)

        logging.info("Documents generated successfully from JSON data")
        return documents

    except Exception as e:
        logging.error(f"Error while generating documents from JSON: {e} trace_back:{traceback.format_exc()}")
        raise Exception(f"Error: {e}")


# Function to process the embedding of the hardcoded or file-based JSON data
def process_embeddings(json_data, collection):
    try:
        # Generate the documents (each containing a Q/A pair)
        docs = generate_documents_from_json(json_data)

        # Check if a vector store already exists for this UUID and remove it
        if collection in VECTOR_STORES.keys():
            vector_store = VECTOR_STORES.get(collection)
            del vector_store

        # Create a new vector store for this UUID and upload the generated documents
        collection_name = f"temp_{collection}"
        qdrant = Qdrant_DB(embedding_model, collection_name)
        vector_store = qdrant.upload_vectors(docs)


        VECTOR_STORES.update({collection: qdrant})

        logging.info(f"Data for {collection} embedded successfully")

    except Exception as e:
        logging.error(f"Error occurred during embedding process: {e}\n{traceback.format_exc()}")



process_embeddings(json_data, collection_id)


app = FastAPI()
origins = ["*"]  # Update based on your specific needs

# Add CORS middleware to the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],  # Limited to POST requests
    allow_headers=["Authorization", "Content-Type"],
)

logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', force=True)

@app.post("/query/")
async def query_endpoint(query: str):
    """
    Executes a query against a vector store identified by UUID.

    Args:
    - query: A string containing the query.
    - uuid: Header identifier for the associated vector store.

    Returns:
    - Response from LLM.
    """
    vector_store = VECTOR_STORES.get(collection_id)
    logging.info(msg=f"vector store : {vector_store}")
    if vector_store is not None:
        try:
            print("Query Endpoint")
            result = gpt_chain(vector_store.vector_store,query)
            return result

        except Exception as e:
            logging.error( f"Error occured while Quering{e} trace_back:{traceback.format_exc()}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Something Went Wrong! Please Try Again.")

    else:
        error_msg = "Vector store Not Created"
        logging.error(error_msg)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail=error_msg)


@app.get("/")
async def health_check_enpoint():
    """
    Provides a health check message for the service.

    Returns:
    - Response: A response object with the content "Hi, I am Healthy RAG".
    """
    return Response(content="Hi, I am Healthy RAG",status_code=status.HTTP_200_OK)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
