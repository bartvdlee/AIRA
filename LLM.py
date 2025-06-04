"""
LLM.py - AI-powered Automated Impact on Rights Assessment (AIRA) Pipeline
This module provides functions to initialize and use language models for conducting
automated fundamental rights impact assessments of AI systems. It provides a full
pipeline for generating comprehensive reports analyzing potential harms, human rights
impacts, and mitigation measures for AI deployment scenarios.
Main components:
- init_llm: Initialize the language model with API authentication
- full_AFRIA: Execute the complete AFRIA pipeline with checkpointing capabilities
The AFRIA pipeline consists of 8 stages:
1. Scenario improvement
2. Stakeholder identification
3. Vignette generation
4. Harm analysis
5. Human rights impact assessment
6. Mitigation measures
7. Severity assessment
8. Likelihood and confidence evaluation
The module supports resuming assessments from checkpoints and generates structured
output in CSV format containing all analysis results.
Example usage:
    full_AFRIA("hiring_ai_report", llm, resume=True)
"""

import sys
import getpass
import os
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
import pandas as pd
import ast
from typing import Optional
import requests
from datasets import load_dataset # type: ignore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import faiss # type: ignore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langgraph.graph.state import CompiledStateGraph
from langchain_community.document_loaders import PyPDFLoader
from tenacity import retry, wait_random_exponential
import asyncio
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Local imports
from LLM_functions import improve_scenario, generate_stakeholders, generate_vignettes, generate_harm, generate_human_rights_impact, generate_mitigation, generate_severity, generate_likelihood, generate_irreversibility
import ask_confirmation
from data_store import DataManager



def get_available_models() -> str:
    """
    Lists all the available models by querying the LiteLLM API.
    The function checks if the API key is already set in the environment variables and asks the user to provide if that is not the case.
    It returns the model name as a string.

    Args:
        None
    Returns:
        str: The name of the selected model.
    """

    # Check if the LiteLLM API key is set in the environment variables
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for LiteLLM: ")

    url = "http://localhost:4000/v1/models"
    headers = {
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
    }

    # Discover all the available models
    response = requests.get(url, headers=headers)
    data = response.json()
    
    # Extract just the model IDs from the response
    model_ids: list[str] = [str(model["id"]) for model in data["data"]]

    # Create a dictionary with the model IDs and their corresponding index numbers
    model_dict: dict[int, str] = {index_no: model for index_no, model in enumerate(model_ids, start=1)}

    # Prints all the available models
    print('Available models:')
    for key, model in model_dict.items():
        print(f'{key}. {model}')
    print()

    # Asks the user to input a model and only continue when the user inputs a valid option
    model_choice = 0
    while model_choice not in model_dict.keys():
        model_choice = input('Please select one of the available models: ')

        if model_choice.isdigit():
            try:
                model_choice = int(model_choice)
            except ValueError:
                print('Input is not a number')

            if model_choice not in model_dict.keys():
                print('The provided number is not a valid option')
        else:
            print('Please input a number')

    model = str(model_dict[int(model_choice)])

    return model



def init_llm(model_name: str = 'mistral/mistral-small-latest') -> BaseChatModel:
    """
    Initializes the LLM specified by `model_name`. It uses liteLLM as backend.
    The function checks if the API key is already set in the environment variables.
    If not, it prompts the user to enter the API key.
    """
    # Check if the LiteLLM API key is set in the environment variables
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for LiteLLM: ")
    
    # Initialize the LLM with the specified model name
    llm = init_chat_model(model_name, model_provider="openai", base_url='http://localhost:4000')
    
    return llm


def init_embeddings(model_name: str = 'text-embedding-005') -> OpenAIEmbeddings:
    """
    Initializes the embeddings model specified by `model_name`. It uses liteLLM as backend.
    The function checks if the API key is already set in the environment variables.
    If not, it prompts the user to enter the API key.
    """
    # Check if the LiteLLM API key is set in the environment variables
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for LiteLLM: ")

    # Initialize the embeddings model with the specified model name
    embeddings = OpenAIEmbeddings(model=model_name, base_url='http://localhost:4000')
    
    return embeddings


def save_vector_store(vector_store: FAISS, folder_path: str ="vector_store") -> None:
    """
    Save the FAISS vector store to disk.
    
    Args:
        vector_store: The FAISS vector store to save.
        folder_path: The folder path to save the vector store to.
    Returns:
        None
    """
    os.makedirs(folder_path, exist_ok=True)
    vector_store.save_local(folder_path)
    print(f"Vector store saved to {folder_path}")


def load_vector_store(embedding_function: OpenAIEmbeddings, folder_path: str ="vector_store"):
    """
    Load the FAISS vector store from disk.
    
    Args:
        embedding_function: The embedding function to use with the vector store.
        folder_path: The folder path to load the vector store from.
    
    Returns:
        The loaded FAISS vector store, or None if it doesn't exist.
    """
    if os.path.exists(folder_path):
        print(f"Loading vector store from {folder_path}")
        return FAISS.load_local(folder_path, embedding_function, allow_dangerous_deserialization=True)
    return None


@retry(wait=wait_random_exponential(multiplier=1, max=60))
def process_batch(vector_store: FAISS, batch_docs: list[Document], batch_uuids: Optional[list[str]] = None) -> FAISS:
    """
    Process a batch of documents and add them to the vector store.
    This function is decorated with retry to handle transient errors.

    Args:
        vector_store: The FAISS vector store to add documents to.
        batch_docs: The batch of documents to add.
        batch_uuids: The UUIDs of the documents.
    Returns:
        The updated vector store.
    Raises:
        Exception: If the batch fails after multiple retries.
    """
    if batch_uuids is None:
        return vector_store.add_documents(documents=batch_docs) # type: ignore
    else:
        return vector_store.add_documents(documents=batch_docs, uuids=batch_uuids) # type: ignore


async def add_documents(vector_store: FAISS, documents: list[Document], uuids: Optional[list[str]] = None) -> None:
    """
    Add documents to the vector store in batches to avoid payload size limits.
    Uses async.
    
    Args:
        vector_store: The FAISS vector store to add documents to.
        documents: The documents to add.
        uuids: The UUIDs of the documents.
    Returns:
        None
    Raises:
        Exception: If the batch fails after multiple retries.
    """
    # Batch size can be changed according to the context size and rate limits of model used
    batch_size = 25
    total_batches = (len(documents) - 1) // batch_size + 1
    processed_batches: set[int] = set()

    for i in range(0, len(documents), batch_size):
        batch_num = i // batch_size + 1
        
        # Skip already processed batches
        if batch_num in processed_batches:
            continue
            
        batch_end = min(i + batch_size, len(documents))
        batch_docs = documents[i:batch_end]
        if not uuids is None:
            batch_uuids: list[str] = uuids[i:batch_end]
        
        print(f"Adding document batch {batch_num}/{total_batches} to vector store...")
        
        try:
            # Call the synchronous function with retry
            if uuids is None:
                process_batch(vector_store, batch_docs)
            else:
                process_batch(vector_store, batch_docs, batch_uuids) # type: ignore

            processed_batches.add(batch_num)
        except Exception as e:
            print(f"Failed to add batch {batch_num} after multiple retries: {e}")
            # Continue with next batch
            await asyncio.sleep(1)  # Brief pause before continuing


def setup_RAG(llm: BaseChatModel) -> CompiledStateGraph:
    """
    Sets up the ECHR RAG (Retrieval-Augmented Generation) system.
    This will load the ECHR data and feed it into the LLM.
    It also adds the ECHR text to the documents.
    The function returns a compiled state graph for the RAG system.

    Args:
        llm (BaseChatModel): The language model to use for the RAG system.
    Returns:
        CompiledStateGraph: The compiled state graph for the RAG system.
    """
    # Initialize the embeddings model
    embeddings = init_embeddings()

    # Try to load existing vector store
    vector_store = load_vector_store(embeddings)

    # If the vector store doesn't exist, create a new one
    if vector_store is None:
        print("No existing vector store found. Creating a new one.")

        # Load the ECHR dataset
        ds = load_dataset("ecthr_cases", "violation-prediction")

        # Load the article translation dictionary (from the ECHR dataset)
        article_translation = {
        # "1": "Obligation to respect Human Rights",
        "2": "Right to life",
        "3": "Prohibition of torture",
        "4": "Prohibition of slavery and forced labour",
        "5": "Right to liberty and security",
        "6": "Right to a fair trial",
        "7": "No punishment without law",
        "8": "Right to respect for private and family life",
        "9": "Freedom of thought, conscience and religion",
        "10": "Freedom of expression",
        "11": "Freedom of assembly and association",
        "12": "Right to marry",
        "13": "Right to an effective remedy",
        "14": "Prohibition of discrimination",
        "15": "Derogation in time of emergency",
        "16": "Restrictions on political activity of aliens",
        "17": "Prohibition of abuse of rights",
        "18": "Limitation on use of restrictions on rights",
        "34": "Individual applications",
        "38": "Examination of the case",
        "39": "Friendly settlements",
        "46": "Binding force and execution of judgments",
        "P1-1": "Protection of property",
        "P1-2": "Right to education",
        "P1-3": "Right to free elections",
        "P3-1": "Right to free elections",
        "P4-1": "Prohibition of imprisonment for debt",
        "P4-2": "Freedom of movement",
        "P4-3": "Prohibition of expulsion of nationals",
        "P4-4": "Prohibition of collective expulsion of aliens",
        "P6-1": "Abolition of the death penalty",
        "P6-2": "Death penalty in time of war",
        "P6-3": "Prohibition of derogations",
        "P7-1": "Procedural safeguards relating to expulsion of aliens",
        "P7-2": "Right of appeal in criminal matters",
        "P7-3": "Compensation for wrongful conviction",
        "P7-4": "Right not to be tried or punished twice",
        "P7-5": "Equality between spouses",
        "P12-1": "General prohibition of discrimination",
        "P13-1": "Abolition of the death penalty",
        "P13-2": "Prohibition of derogations",
        "P13-3": "Prohibition of reservations"
        }

        # Add the European Convention on Human Rights (ECHR) text to the documents
        documents: list[Document] = []
        for doc in PyPDFLoader("ECHR.pdf").lazy_load():
            # Add the document to the list
            documents.append(doc)
        print("Successfully loaded the ECHR text.")

        # Get the number of documents loaded from the ECHR text for later use
        documents_echr_len = len(documents)

        # Add the ECHR dataset to the documents
        case_no = 0
        for item in ds['train']: # type: ignore
            """
            Creates Document objects for every case in the ECHR dataset.
            Every fact in the case is a separate Document and contains metadata about the case.
            The metadata includes the case no (does not represent real-world no), verdict, and whether fact was referenced by judge in verdict.
            This way the labels and silver_rationales columns from the dataset are incorporated into the fact Documents.
            Only the facts of the case will put into Documents.
            """
            # Increment the case number (does not represent real-world case number)
            case_no += 1

            # Create a comma-separated string of the violated articles for this case
            verdict: str = ', '.join([f"Article {article} ({article_translation[article]})" if len(item['labels']) > 0 else "No articles violated" for article in item['labels']]) # type: ignore
            
            # Keep track of the list index of the facts to determine if the fact was referenced in the verdict
            fact_index = 0

            for fact in item['facts']: # type: ignore
                # Create a Document object for every fact in the case and include metadata about the case
                documents.append(Document(page_content=fact, metadata={ # type: ignore
                    "case_no": case_no,
                    "verdict": verdict,
                    "fact_is_referenced_in_verdict": True if fact_index in item['silver_rationales'] else False # type: ignore
                }))

                # Increment the fact index
                fact_index += 1

        # Print the number of documents loaded from the ECHR dataset minus the ECHR text
        print(f"Loaded {len(documents) - documents_echr_len} documents from the ECHR dataset.")

        # Create vector store
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
        vector_store = FAISS(embedding_function=embeddings, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})

        # Add the documents to the vector store in batches to avoid size and rate limits, use async and retry
        asyncio.run(add_documents(vector_store, documents))

        # Save the vector store to disk
        save_vector_store(vector_store)
    else:
        print("Using existing vector store.")

    # Creates the Graph for the RAG system
    # It mimics the Graph from the generate_human_rights_impact.py file as closely as possible

    # To be able to mimic the Graph from the generate_human_rights_impact.py file, we need to define the HumanRightGeneration class for format instructions
    class HumanRightGeneration(BaseModel):
        """Generation of lists of affected human rights and argumentations."""

        human_rights: list[str] = Field(description="List of human rights (as defined by the European Convention on Human Rights) affected by the AI system for the stakeholder")
        argumentations: list[str] = Field(description="List of argumentations for each human right affected by the AI system for the stakeholder. The argumentation should explain why the human right is affected by the AI system and how it is violated, cite sources used.")
    
    # Define the output parser for the human rights generation
    parser = PydanticOutputParser(pydantic_object=HumanRightGeneration)

    # Now we define the prompt (where we added one line for RAG) and use the format instructions from the parser
    prompt = ChatPromptTemplate.from_messages( # type: ignore
    [
        ("system", "You are an expert in AI ethics and you are asked to evaluate the human rights affected by certain harms."
                   "Please identify the human rights (as defined by the European Convention on Human Rights) affected by the given harm for the specified stakeholder."
                   "For each right, provide an argumentation explaining why it is affected."
                   "Use the provided context (ECHR text and relevant case law) to answer the question:\n\n{context}\n\n" # Added this line for RAG
                   "Wrap the output in `json` tags\n{format_instructions}"
                   "Your response must be valid JSON"),
        ("human", "{question}"),
    ]).partial(format_instructions=parser.get_format_instructions())

    # Define state for application
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    # Define application steps
    def retrieve(state: State): # type: ignore
        retrieved_docs = vector_store.similarity_search(state["question"], k=10) # type: ignore

        # Group documents by case number
        cases = {}
        for doc in retrieved_docs:
            case_no = doc.metadata.get("case_no") # type: ignore

            if case_no not in cases:
                cases[case_no] = {
                    'verdict' : doc.metadata.get("verdict", "No verdict available"), # type: ignore
                    'facts': [],
                    'referenced_facts': []
                }
            
            # Seperate facts based on whether they were referenced in the verdict
            if doc.metadata.get("fact_is_referenced_in_verdict", False): # type: ignore
                cases[case_no]['referenced_facts'].append(doc) # type: ignore
            else:
                cases[case_no]['facts'].append(doc) # type: ignore
        
        # Prioritize referenced facts in the context
        prioritized_docs = []
        for case_no, case_data in cases.items(): # type: ignore
            # Add referenced facts first
            prioritized_docs.extend(case_data['referenced_facts']) # type: ignore
            # Then add other facts
            prioritized_docs.extend(case_data['facts']) # type: ignore
        
        return {"context": prioritized_docs} # type: ignore


    def generate(state: State): # type: ignore
        # Group context documents by case number for presentation
        cases = {}
        echr_docs = []

        for doc in state["context"]:
            case_no = doc.metadata.get("case_no") # type: ignore
            if case_no is not None:
                if case_no not in cases:
                    cases[case_no] = {
                        "verdict": doc.metadata.get("verdict", "No verdict available"), # type: ignore
                        "facts": [],
                        "referenced_facts": []
                    }
                
                if doc.metadata.get("fact_is_referenced_in_verdict", False): # type: ignore
                    cases[case_no]["referenced_facts"].append(doc) # type: ignore
                else:
                    cases[case_no]["facts"].append(doc) # type: ignore
            else:
                # If no case number, treat as ECHR document
                echr_docs.append(doc) # type: ignore

        structured_context: list[str] = []

        # Add ECHR treaty text first if available
        if echr_docs:
            structured_context.append("Relevant ECHR treaty provisions:") # type: ignore
            structured_context.extend(echr_docs) # type: ignore

        # Add cases with their verdicts and facts
        if cases:
            structured_context.append("\nRelevant ECHR case law:") # type: ignore
            for case_no, case_data in cases.items(): # type: ignore
                structured_context.append(f"\nCase {case_no} - Verdict: {case_data.get('verdict', 'No articles violated')}") # type: ignore

                if case_data['referenced_facts']:
                    structured_context.append("Key facts (referenced in judgment):") # type: ignore
                    for fact in case_data["referenced_facts"]: # type: ignore
                        structured_context.append(f"- {fact}") # type: ignore
                    
                if case_data['facts']:
                    structured_context.append("Additional case Facts:") # type: ignore
                    for fact in case_data["facts"]: # type: ignore
                        structured_context.append(f"- {fact}") # type: ignore

            # Add spacing between cases
            structured_context.append("\n") # type: ignore

        # docs_content = "\n".join([str(doc) for doc in structured_context])
        messages = prompt.invoke({"question": state["question"], "context": structured_context}) # type: ignore
        response = llm.invoke(messages)
        return {"answer": response.content} # type: ignore

    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate]) # type: ignore
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile() # type: ignore


def full_AFRIA(report_name: str, llm: BaseChatModel = init_llm(), resume: bool = True, start_from: Optional[str] = None) -> None:
    '''
    CLI tool to run the full AFRIA process.
    The function does not return any value.
    It takes the report name, the LLM to use, a boolean to resume from the last checkpoint and an optional starting stage as input.
    The function runs the full AFRIA process and saves the results in a CSV file.

    The `resume` parameter acts as a master switch for the entire checkpointing system.
    If `resume` is set to True, the function will check for existing checkpoints and load data from them.
    If `resume` is set to False, the function will ignore any existing checkpoints and start fresh.

    The `start_from` parameter allows the user to specify a stage from which to start the process.
    If `start_from` is provided, the function will start from that stage, when `resume` is True it uses checkpoints until the `start_from` stage is reached.
    If `start_from` is not provided, the function will start from the first stage.

    The stages are as follows:
    1. Improved Scenario
    2. Stakeholders
    3. Vignettes
    4. Harms
    5. Human Rights Impact
    6. Mitigation Measures
    7. Severity
    8. Likelihood
    9. Irreversibility

    The function generates the data for each stage using the LLM and saves it in a CSV file.
    The CSV file is saved in the reports folder with the name of the report.
    The CSV file contains the following columns:
    - Category: The category of the stakeholder (direct or indirect)
    - Stakeholder: The name of the stakeholder
    - Problematic Behaviour: The name of the problematic behaviour
    - Vignette: The vignette of the stakeholder
    - Harm: The harm of the stakeholder
    - Human Rights Impact: The human rights impact of the stakeholder
    - Human Rights Argumentation: The human rights argumentation of the stakeholder
    - Mitigation Measures: The mitigation measures of the stakeholder
    - Severity Level: The severity level of the stakeholder
    - Severity Justification: The severity justification of the stakeholder
    - Likelihood Level: The likelihood level of the stakeholder
    - Likelihood Justification: The likelihood justification of the stakeholder
    - Irreversibility Level: The irreversibility level of the stakeholder
    - Irreversibility Justification: The irreversibility justification of the stakeholder

    Args:
        report_name (str): The name of the report to run.
        llm (BaseChatModel): The LLM to use for the process. Default is the default LLM.
        resume (bool): Whether to resume from the last checkpoint. Default is True.
        start_from (Optional[str]): The stage to start from. Default is None.
    Returns:
        None
    '''
    # Initialize data manager
    dm = DataManager(report_name)

    # Load the report information from the CSV file
    try:
        report_info: pd.DataFrame = pd.read_csv(f"reports/{report_name}/info.csv", index_col=0) # type: ignore

        if 'scenario' not in report_info.index or report_info.loc['scenario'].empty: # type: ignore
            raise ValueError("The 'scenario' row is missing or empty in the report.")
        scenario = str(report_info.loc['scenario'].iloc[0]) # type: ignore

        user_specified_stakeholders: list[str] = ast.literal_eval(str(report_info.loc['user_specified_stakeholders'].iloc[0])) # type: ignore
        problematic_behaviours: list[str] = ast.literal_eval(str(report_info.loc['problematic_behaviours'].iloc[0])) # type: ignore
        # harm_dimensions are not implemented

        print(f"Successfully loaded report {report_name}")
    except FileNotFoundError:
        print(f"Report {report_name} not found.")
        sys.exit(1)
    
    # Assert that the report information is not empty
    if len(scenario) == 0:
        raise ValueError("Scenario is empty")
    if len(user_specified_stakeholders) == 0:
        raise ValueError("User-specified stakeholders are empty")
    if len(problematic_behaviours) == 0:
        raise ValueError("Problematic behaviours are empty")

    # Stages in order of execution
    stages = [
        "improved_scenario", 
        "stakeholders", 
        "vignettes", 
        "harms", 
        "human_rights", 
        "mitigations",
        "severity",
        "likelihood",
        "irreversibility"
    ]

    # Determine where to start
    start_stage_idx = 0
    if resume and not start_from:
        # Find the last completed stage
        available_checkpoints = dm.list_data()
        for i, stage in reversed(list(enumerate(stages))):
            if stage in available_checkpoints:
                start_stage_idx = i
                break
    elif start_from:
        try:
            start_stage_idx = stages.index(start_from)
        except ValueError:
            print(f"Stage {start_from} not found. Available stages: {stages}")
            sys.exit(1)


    # SCENARIO IMPROVEMENT
    if start_stage_idx <= 0:
        # Generate an improved scenario using the LLM
        improved_scenario = improve_scenario.improve(scenario, llm)
        dm.save_data("improved_scenario", {"original": scenario, "improved": improved_scenario})
        
        print("The LLM-based tool has suggested an improved scenario.")
        print(f"Original scenario: {scenario}")
        print(f"Improved scenario: {improved_scenario}")

        # Ask the user if they want to use the improved scenario
        if ask_confirmation.closed('Do you want to use the improved scenario (y/n)?'):
            scenario = improved_scenario
            print("Successfully updated the scenario.")
        else:
            print("Keeping the original scenario.")
    else:
        # Load from checkpoint
        checkpoint_data = dm.load_data("improved_scenario")
        if checkpoint_data and "improved" in checkpoint_data:
            print(f"Original scenario: {checkpoint_data["original"]}")
            print(f"Improved scenario: {checkpoint_data["improved"]}")

            if ask_confirmation.closed('Use improved scenario from checkpoint? (y/n)'):
                scenario = checkpoint_data["improved"]
            else:
                scenario = checkpoint_data["original"]

    
    # STAKEHOLDERS GENERATION
    if start_stage_idx < 1:
        # Generate stakeholders using the LLM
        print('Generating stakeholders...')

        stakeholders = generate_stakeholders.generate(scenario, user_specified_stakeholders, llm)
        dm.save_data("stakeholders", stakeholders)
    else:
        # Load from checkpoint
        stakeholders = dm.load_data("stakeholders")
    
    # Create a list of all stakeholders for later use
    all_stakeholders: list[str] = stakeholders['direct_stakeholders'] + stakeholders['indirect_stakeholders']
    print('Successfully loaded/generated stakeholders.')


    # VIGNETTES GENERATION
    if start_stage_idx < 2:
        # Generate vignettes using the LLM
        print('Generating vignettes...')

        vignettes: dict[str, list[str]] = {}
        for stakeholder in all_stakeholders:
            vignettes[stakeholder] = generate_vignettes.generate(scenario, stakeholder, problematic_behaviours, llm)

        # Save the data using the DataManager
        dm.save_data("vignettes", vignettes)

        print('Successfully generated vignettes.')
    else:
        # Load from checkpoint
        vignettes = dm.load_data("vignettes")
        print('Successfully loaded vignettes from checkpoint.')


    # HARMS GENERATION
    if start_stage_idx < 3:
        # Generate harms for the vignettes using the LLM
        print('Generating harms for the vignettes...')

        harms: dict[str, list[str]] = {}
        for stakeholder in all_stakeholders:
            harms[stakeholder] = generate_harm.generate(vignettes[stakeholder], stakeholder, llm)
        
        # Save the data using the DataManager
        dm.save_data("harms", harms)

        print('Successfully generated harms for the vignettes.')
    else:
        # Load from checkpoint
        harms = dm.load_data("harms")
        print('Successfully loaded harms from checkpoint.')


    # HUMAN RIGHTS IMPACT GENERATION
    if start_stage_idx < 4:
        # Generate human rights impact using the LLM
        print('Generating human rights impacts for the harms...')

        human_rights: dict[str, list[list[str]]] = {}
        argumentations: dict[str, list[list[str]]] = {}
        for stakeholder in all_stakeholders:
            data = generate_human_rights_impact.generate(harms[stakeholder], stakeholder, llm)

            human_rights[stakeholder] = data["human_rights"]
            argumentations[stakeholder] = data["argumentations"]

        # Save the data using the DataManager
        dm.save_data("human_rights", {
            "human_rights": human_rights,
            "argumentations": argumentations
        })
        
        print('Successfully generated human rights impact for the harms.')
    else:
        # Load from checkpoint
        checkpoint_data = dm.load_data("human_rights")

        human_rights = checkpoint_data["human_rights"]
        argumentations = checkpoint_data["argumentations"]

        print('Successfully loaded human rights impact from checkpoint.')


    # MITIGATION MEASURES GENERATION
    if start_stage_idx < 5:
        # Generate mitigation measures using the LLM
        print('Generating mitigation measures for the harms...')

        mitigation_measures: dict[str, list[list[str]]] = {}
        for stakeholder in all_stakeholders:
            mitigation_measures[stakeholder] = generate_mitigation.generate(stakeholder, harms[stakeholder], human_rights[stakeholder], llm)

        # Save the data using the DataManager
        dm.save_data("mitigations", mitigation_measures)

        print('Successfully generated mitigation measures.')
    else:
        # Load from checkpoint
        mitigation_measures = dm.load_data("mitigations")
        print('Successfully loaded mitigation measures from checkpoint.')


    # SEVERITY GENERATION
    if start_stage_idx < 6:
        # Generate severity levels using the LLM
        print('Generating severity levels for the harms...')

        severity_levels: dict[str, list[list[str]]] = {}
        severity_justifications: dict[str, list[list[str]]] = {}

        for stakeholder in all_stakeholders:
            data = generate_severity.generate(harms[stakeholder], human_rights[stakeholder], llm)

            severity_levels[stakeholder] = data["severity_levels"]
            severity_justifications[stakeholder] = data["justifications"]

        # Save the data using the DataManager
        dm.save_data("severity", {
            "severity_levels": severity_levels,
            "justifications": severity_justifications
        })

        print('Successfully generated severity levels.')
    else:
        # Load from checkpoint
        checkpoint_data = dm.load_data("severity")

        severity_levels = checkpoint_data["severity_levels"]
        severity_justifications = checkpoint_data["justifications"]

        print('Successfully loaded severity levels from checkpoint.')


    # LIKELIHOOD GENERATION
    if start_stage_idx < 7:
        # Generate likelihood and confidence levels using the LLM
        print('Generating likelihood and confidence levels for the harms...')

        likelihood_levels: dict[str, list[list[str]]] = {}
        likelihood_justifications: dict[str, list[list[str]]] = {}
        for stakeholder in all_stakeholders:
            data = generate_likelihood.generate(harms[stakeholder], human_rights[stakeholder], llm)

            likelihood_levels[stakeholder] = data["likelihood_levels"]
            likelihood_justifications[stakeholder] = data["justifications"]

        # Save the data using the DataManager
        dm.save_data("likelihood", {
            "likelihood_levels": likelihood_levels,
            "likelihood_justifications": likelihood_justifications
        })

        print('Successfully generated likelihood and confidence levels.')
    else:
        # Load from checkpoint
        data = dm.load_data("likelihood")
        likelihood_levels = data["likelihood_levels"]
        likelihood_justifications = data["likelihood_justifications"]

        print('Successfully loaded likelihood and confidence levels from checkpoint.')


    # IRREVERSIBILITY GENERATION
    if start_stage_idx < 8:
        # Generate irreversibility levels using the LLM
        print('Generating irreversibility levels for the harms...')

        irreversibility_levels: dict[str, list[list[str]]] = {}
        irreversibility_justifications: dict[str, list[list[str]]] = {}

        for stakeholder in all_stakeholders:
            data = generate_irreversibility.generate(harms[stakeholder], human_rights[stakeholder], llm)

            irreversibility_levels[stakeholder] = data["irreversibility_levels"]
            irreversibility_justifications[stakeholder] = data["justifications"]

        # Save the data using the DataManager
        dm.save_data("irreversibility", {
            "irreversibility_levels": irreversibility_levels,
            "irreversibility_justifications": irreversibility_justifications
        })

        print('Successfully generated irreversibility levels.')
    else:
        # Load from checkpoint
        data = dm.load_data("irreversibility")
        irreversibility_levels = data["irreversibility_levels"]
        irreversibility_justifications = data["irreversibility_justifications"]

        print('Successfully loaded irreversibility levels from checkpoint.')

    # Next step: save it in a dataframe
    # First we create the multi-index of the dataframe

    # Create a list with the categories (direct/indirect) of all the stakeholders
    category = (['Direct stakeholders'] * len(stakeholders['direct_stakeholders'])) + (['Indirect stakeholders'] * len(stakeholders['indirect_stakeholders']))
    
    # Create a list with a short name for the problematic behaviours to keep the dataframe tidy
    try:
        problematic_behaviours_short = [behavior.split(' (')[0] for behavior in problematic_behaviours]
    except Exception as e:
        print(f"Error shortening problematic behaviours: {e}")
        problematic_behaviours_short = problematic_behaviours

    # Create expanded arrays for the MultiIndex
    # For each stakeholder, include all problematic behaviors
    expanded_categories: list[str] = []
    expanded_stakeholders: list[str] = []
    expanded_behaviors: list[str] = []

    for i, stakeholder in enumerate(all_stakeholders):
        for behavior in problematic_behaviours_short:
            expanded_categories.append(category[i])
            expanded_stakeholders.append(stakeholder)
            expanded_behaviors.append(behavior)

    # Create the multi-index with three levels
    index = pd.MultiIndex.from_arrays( # type: ignore
        [expanded_categories, expanded_stakeholders, expanded_behaviors], 
        names=('Category', 'Stakeholder', 'Problematic Behavior')
    )

    # Now that we have the multi-index, we can create the DataFrame
    rows: list[dict[str, str | list[str | None]]] = []
    for i, stakeholder in enumerate(all_stakeholders):
        for j, behavior in enumerate(problematic_behaviours_short):
            # Ensure all lists for a given stakeholder/behavior have the same length
            max_length = max(len(human_rights[stakeholder][j]), 
                            len(argumentations[stakeholder][j]),
                            len(mitigation_measures[stakeholder][j]),
                            len(severity_levels[stakeholder][j]),
                            len(severity_justifications[stakeholder][j]),
                            len(likelihood_levels[stakeholder][j]),
                            len(likelihood_justifications[stakeholder][j]),
                            len(irreversibility_levels[stakeholder][j]),
                            len(irreversibility_justifications[stakeholder][j]))


            # Then pad shorter lists with None or empty strings
            padded_human_rights = human_rights[stakeholder][j] + [None] * (max_length - len(human_rights[stakeholder][j]))
            padded_human_right_argumentations = argumentations[stakeholder][j] + [None] * (max_length - len(argumentations[stakeholder][j]))
            padded_mitigation = mitigation_measures[stakeholder][j] + [None] * (max_length - len(mitigation_measures[stakeholder][j]))
            padded_severity = severity_levels[stakeholder][j] + [None] * (max_length - len(severity_levels[stakeholder][j]))
            padded_severity_justifications = severity_justifications[stakeholder][j] + [None] * (max_length - len(severity_justifications[stakeholder][j]))
            padded_likelihood = likelihood_levels[stakeholder][j] + [None] * (max_length - len(likelihood_levels[stakeholder][j]))
            padded_likelihood_justifications = likelihood_justifications[stakeholder][j] + [None] * (max_length - len(likelihood_justifications[stakeholder][j]))
            padded_irreversibility = irreversibility_levels[stakeholder][j] + [None] * (max_length - len(irreversibility_levels[stakeholder][j]))
            padded_irreversibility_justifications = irreversibility_justifications[stakeholder][j] + [None] * (max_length - len(irreversibility_justifications[stakeholder][j]))


            rows.append({
                'Vignette': vignettes[stakeholder][j],
                'Harm': harms[stakeholder][j],
                'Human rights impact': padded_human_rights,
                'Human rights argumentation': padded_human_right_argumentations,
                'Mitigation measures': padded_mitigation,
                'Severity level': padded_severity,
                'Severity justification': padded_severity_justifications,
                'Likelihood level': padded_likelihood,
                'Likelihood justification': padded_likelihood_justifications,
                'Irreversibility level': padded_irreversibility,
                'Irreversibility justification': padded_irreversibility_justifications,
            })

    df = pd.DataFrame.from_records(rows, index=index) # type: ignore

    # Explode the DataFrame to split the human rights and subsequent columns into individual rows
    df = df.explode(['Human rights impact', 'Human rights argumentation', 'Mitigation measures', 'Severity level', 'Severity justification', 'Likelihood level', 'Likelihood justification', 'Irreversibility level', 'Irreversibility justification']) # type: ignore
    df = df.dropna(subset=['Human rights impact'], axis=0) # type: ignore
   
    print('Successfully generated the report data.')
    df.to_csv(f'reports/{report_name}/output.csv')



if __name__ == "__main__":

    # Small CLI environment for quickly testing the RAG

    if ask_confirmation.closed('Do you want to use a custom LLM (default: Mistral Small) (y/n)?'):
        llm = init_llm(get_available_models())
        print('Successfully loaded the custom LLM.', end='\n\n')
    else:
        print('Using the default LLM.', end='\n\n')
        llm = init_llm()

    graph = setup_RAG(llm)

    # with open("rag_graph.png", "wb") as f:
    #     f.write(graph.get_graph().draw_mermaid_png())
    # print("Graph image saved as rag_graph.png")

    thread_id = "12345678"

    # result = graph.invoke({"question": "What human rights of the stakeholder 'The applicant' are affected by the following harms, limit your answer to only provide the most important rights: Imagine you are an applicant who receives an invitation for an interview for a job you applied for. You are initially excited, but as you review the job requirements, you realize that the AI system has incorrectly identified you as a suitable candidate. You lack the necessary skills and experience for the role, leading to frustration and confusion. This misidentification can harm your self-esteem, waste your time, and potentially damage your credibility if you proceed with the interview and your inadequacies become apparent."},
    #                       config={"configurable": {"thread_id": thread_id}})

    print("\n\n\n\n")
    result1 = graph.invoke({"question" : input('Enter prompt: ')}, config={"configurable" : {"thread_id" : thread_id}})

    print(f'Context: {result1["context"]}\n\n')
    print(f'Answer: {result1["answer"]}')

    print("\n\n\n\n")
    result2 = graph.invoke({"question" : input('Enter prompt: ')}, config={"configurable" : {"thread_id" : thread_id}})

    print(f'Context: {result2["context"]}\n\n')
    print(f'Answer: {result2["answer"]}')

    print("\n\n\n\n")
    result3 = graph.invoke({"question" : input('Enter prompt: ')}, config={"configurable" : {"thread_id" : thread_id}})

    print(f'Context: {result3["context"]}\n\n')
    print(f'Answer: {result3["answer"]}')

    print("\n\n\n\n")
    result4 = graph.invoke({"question" : input('Enter prompt: ')}, config={"configurable" : {"thread_id" : thread_id}})

    print(f'Context: {result4["context"]}\n\n')
    print(f'Answer: {result4["answer"]}')

    print("\n\n\n\n")
    result5 = graph.invoke({"question" : input('Enter prompt: ')}, config={"configurable" : {"thread_id" : thread_id}})

    print(f'Context: {result5["context"]}\n\n')
    print(f'Answer: {result5["answer"]}')