"""
Human Rights Impact Generation Module
This module provides functionality to generate potential human rights impacts
based on identified harms to specific stakeholders. It leverages language models
to identify which articles of the European Convention on Human Rights may be
affected by specific harms.
The module uses LangGraph for workflow management and LangChain for language model interactions.

Functions:
    init_workflow(llm): Initialize the LangGraph workflow with a language model.
    call_model(state, llm): Helper function to invoke the language model with a given state.
    generate(harms, stakeholder, llm): Generate lists of human rights affected by each harm.
"""
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Optional, Any
from langchain_core.prompt_values import PromptValue

# local imports
from data_store import DataManager


class HumanRightGeneration(BaseModel):
        """Generation of lists of affected human rights and argumentations."""

        human_rights: list[str] = Field(description="List of human rights (as defined by the European Convention on Human Rights) affected by the AI system for the stakeholder")
        argumentations: list[str] = Field(description="List of argumentations for each human right affected by the AI system for the stakeholder. The argumentation should explain why the human right is affected by the AI system and how it is violated, cite sources used.")


def generate(harms: list[str], stakeholder: str, llm: BaseChatModel | CompiledStateGraph, dm: Optional[DataManager] = None) -> dict[str, list[list[str]]]:
    '''
    Generates a list of human rights affected by the given harms for the specified stakeholder using the provided `llm`.
    The function returns a dictionary with list of lists, where each inner list contains the affected human rights for each harm or the argumentation for the human right.
    It also provides argumentations for each human right, explaining why it is affected by the harm.
    The human rights are generated based on the European Convention on Human Rights.
    The function works for BaseChatModel and CompiledStateGraph (RAG) types.
    
    Args:
        harms (list[str]): A list of harms to consider.
        stakeholder (str): The stakeholder for whom to generate the human rights.
        llm (BaseChatModel | CompiledStateGraph): The language model to use for generating the human rights.
        dm (Optional[DataManager]): An optional DataManager instance for saving context when using RAG.
    Returns:
        dict[str, list[list[str]]]: a dict with keys human_rights and argumentations which contain a list of lists, where each inner list contains the affected human rights for each harm or argumentation.
    '''
    parser = PydanticOutputParser(pydantic_object=HumanRightGeneration)

    def invoke_with_retry(llm: BaseChatModel | CompiledStateGraph, message: PromptValue | str, max_retries: int = 5, RAG: bool = False) -> dict[str, Any]: # type: ignore
        """Helper function to invoke LLM with retry logic for JSON parsing errors."""
        if not RAG:
            for attempt in range(max_retries):
                try:
                    response_data = llm.invoke(message)

                    structured_response = parser.parse(response_data.content) # type: ignore

                    return {'structured_response': structured_response,
                            'response_data': response_data}
                except Exception as e:
                    if attempt < max_retries:
                        print(f"Error parsing response, retrying... ({attempt + 1}/{max_retries})")
                    else:
                        raise e
        else:
            for attempt in range(max_retries):
                try:
                    response_data = llm.invoke({"question" : message}) # type: ignore

                    structured_response = parser.parse(response_data['answer']) # type: ignore

                    return {'structured_response': structured_response,
                        'response_data': response_data}
                except Exception as e:
                    if attempt < max_retries:
                        print(f"Error parsing response, retrying... ({attempt + 1}/{max_retries})")
                    else:
                        raise e

    if isinstance(llm, BaseChatModel):
        # non RAG mode

        prompt = ChatPromptTemplate.from_messages( # type: ignore
        [
            ("system", "You are an expert in AI ethics and you are asked to evaluate the human rights affected by certain harms. "
                       "Please identify the human rights (as defined by the European Convention on Human Rights) affected by the given harm for the specified stakeholder. "
                       "For each right, provide an argumentation explaining why it is affected."
                       "Wrap the output in `json` tags\n{format_instructions}"
                       "Your response must be valid JSON"),
            ("human", "{query}"),
        ]).partial(format_instructions=parser.get_format_instructions())

        human_rights: list[list[str]] = list()
        argumentations: list[list[str]] = list()

        for harm in harms:       
            # Invoke the model with the message
            message = prompt.invoke({"query" : """What human rights (as defined by the European Convention on Human Rights) of the """ # type: ignore
                                        f"""stakeholder '{stakeholder}' are affected by this harm: {harm}
                                        Refrain from mentioning rights and freedoms of other stakeholders, but only focus on the
                                        rights and freedoms of {stakeholder}. Also provide an argumentation for each right.
                                        Give the answer in a JSON format.
                                        Use the following format for providing the human rights: Article [article number] ([article name])."""
            })

            # Invoke the model with the message
            structured_response = invoke_with_retry(llm, message)['structured_response']

            human_rights.append(structured_response.human_rights)
            argumentations.append(structured_response.argumentations)

    elif isinstance(llm, CompiledStateGraph): # type: ignore	
        # RAG mode
        human_rights: list[list[str]] = list()
        argumentations: list[list[str]] = list()
        answer_context: list[str] = list()

        for harm in harms:
            # Invoke the model with the message
            message = f"""What human rights (as defined by the European Convention on Human Rights) of the
                              stakeholder '{stakeholder}' are affected by this harm: {harm}
                              Refrain from mentioning rights and freedoms of other stakeholders, but only focus on the
                              rights and freedoms of {stakeholder}. Also provide an argumentation for each right.
                              Give the answer in a JSON format.
                              Use the following format for providing the human rights: Article [article number] ([article name])."""

            # Invoke the model with the message
            response = invoke_with_retry(llm, message, RAG=True)

            structured_response = response['structured_response']
            response_data = response['response_data']

            human_rights.append(structured_response.human_rights)
            argumentations.append(structured_response.argumentations)
            answer_context.append(str(response_data['context']))
        
        # Save the context to a json file
        if isinstance(dm, DataManager):
            dm.save_data(f"human_rights_context_{stakeholder}".replace(' ', '_'), answer_context)
        else:
            print("DataManager instance not provided. Context not saved.")
    
    else:
        raise TypeError(f"The `llm` parameter must be either a BaseChatModel or a CompiledStateGraph instance, but received type: {type(llm).__name__}.")

    return {"human_rights": human_rights, "argumentations": argumentations}