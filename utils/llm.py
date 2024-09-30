from langchain.chains import RetrievalQA , LLMChain

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import logging
import traceback
import re

load_dotenv()

openai_api_key = os.getenv("openai_api_key")
print(openai_api_key)
similarity_threshold = 0.55  # Define the similarity threshold

try:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
    template = """
    Objectives:
    provide me an efficient answer and Please be consice and to the point
    Instructions:
    If you don't know the answer, just say that you don't know, don't try to make an answer.
    {context}
    Question: {question}
    Helpful Answer:
    """
    template2 = """
    Objectives:
    To check is relevant to the user query or not contextualy.
    Instructions:
    If the answer seems relevant to user query respond with 'yes' and if not respond with 'no':


    Question: {question}
    Answer: {answer}
    Helpful Answer:
    """
except Exception as e:
    error_msg = "Error while Initializing ChatOpenAi: "
    logging.error(f"{error_msg}{e} trace_back:{traceback.format_exc()}")
    raise Exception(f"{error_msg} {e}")




# Assuming `llm` is already initialized (ChatOpenAI model, for instance)

def validate_relevance_with_llm(question, answer):
    """
    Validate if the given answer is relevant to the user query using LLM.

    Parameters:
    - question: The user query.
    - answer: The retrieved answer to validate.

    Returns:
    - Boolean: True if LLM confirms the answer is relevant, False otherwise.
    """
    prompt = PromptTemplate(input_variables=["question", "answer"], template=template2)
    # prompt_text = prompt.format(question=question, answer=answer)

    # Call the LLM with the formatted prompt
    try:
        chain = LLMChain(llm=llm, prompt=prompt)

        response = chain.invoke({"question":question,
                                  "answer":answer})
        logging.info(f"LLM response: {response['text']}")

        # Use regex to check if the LLM response contains "yes" or "no"
        if re.search(r'\byes\b', response['text'], re.IGNORECASE):
            return True
        elif re.search(r'\bno\b', response['text'], re.IGNORECASE):
            return False
        else:
            return False  # Default to "not relevant" if the response is unclear

    except Exception as e:
        logging.error(f"Error during LLM relevance validation: {e}")
        return False


def retrieve_similar_qa(question, vector_store):
    """
    Retrieve the most similar Q/A pair from the stored embeddings (JSON-based).

    Parameters:
    - question: The question string to compare.
    - vector_store: The vector store where embeddings of Q/A pairs are stored.

    Returns:
    - A tuple (most_similar_answer, similarity_score) if relevant, else (None, 0).
    """
    try:
        logging.info("Retrieving similar Q/A pair from the stored embeddings")

        # Get the top result from the retriever, which includes similarity score
        search_results = vector_store.similarity_search_with_score(query=question, k=1)
        logging.info(f"Search results: \n {search_results}\n")

        if search_results:
            most_relevant_doc = search_results[0]  # Get the top result
            similarity_score = most_relevant_doc[1]  # Extract similarity score
            most_similar_answer = most_relevant_doc[0].metadata.get("answer")

            # Use LLM to validate the relevance of the answer
            is_relevant = validate_relevance_with_llm(question, most_similar_answer)

            if is_relevant:
                logging.info(f"Answer is relevant with similarity score: {similarity_score}")
                return most_similar_answer, similarity_score
            else:
                logging.info(f"Answer is not relevant.")
                return None, 0.0

        return None, 0.0  # Return no result if no similar result is found

    except Exception as e:
        logging.error(f"Error during similarity retrieval: {e} trace_back: {traceback.format_exc()}")
        return None, 0.0

def gpt_chain(vector_store, question):
    """
    Retrieves context and generates an answer to a question using GPT.

    Parameters:
    - vector_store: The vectorized data store for context retrieval.
    - question: The question string to be processed.

    Returns:
    - str: Contains the generated answer or the most similar answer from the vector store.
    """

    try:
        logging.info("Checking for a similar Q/A pair in the vector store")
        # Retrieve the most similar answer from the stored embeddings
        most_similar_answer, similarity_score = retrieve_similar_qa(question, vector_store)

        if similarity_score >= similarity_threshold:
            # If the similarity is above the threshold, return the retrieved answer
            logging.info("Returning the retrieved answer directly based on high similarity")
            return most_similar_answer

        # If no high similarity result is found, fall back to the LLM
        logging.info("No highly similar Q/A pair found, falling back to LLM")

        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["question", "context"], template=template)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        result = qa_chain({"query": question})
        logging.info("Q/A Chain Processed Successfully")
        return result["result"]

    except Exception as e:
        error_msg = "Error while generating Q/A Chain: "
        logging.error(f"{error_msg}{e} trace_back:{traceback.format_exc()}")
        raise Exception(f"{error_msg} {e}")
