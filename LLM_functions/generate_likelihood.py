"""
Module for generating likelihood and justification for potential human rights harms.
This module provides functionality to evaluate the likelihood of specific harms 
occurring within the context of human rights using a language model. It returns
structured assessments with both likelihood levels and justifications.
Classes:
    G Pydantic model for structured output representing 
                          likelihood and justification.
Functions:
    generate: Evaluates likelihood and justification for harms in the context 
             of human rights using a language model.
"""

from typing import Optional, Any
from pydantic import BaseModel, Field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.output_parsers import PydanticOutputParser
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver


class LikelihoodGeneration(BaseModel):
    """A Pydantic model representing structured output for likelihood and justification 
    associated with human rights harms. This model includes fields for likelihood and 
    justification."""

    likelihood: str = Field(description="Likelihood of the harm occuring within the context of the given human right")
    justification: str = Field(description="Justification for the choice of the likelihood level")

    def to_dict(self):
        return {
            "likelihood": self.likelihood,
            "justification": self.justification
        }

class EnhancedMessagesState(MessagesState):
    structured_output: Optional[dict[str, Any]]

# Create parser
parser = PydanticOutputParser(pydantic_object=LikelihoodGeneration)

def init_workflow(llm: BaseChatModel, system_prompt: SystemMessage) -> CompiledStateGraph:
    '''
    Initialize the workflow with the state schema and nodes.
    '''
    # Create the workflow
    workflow = StateGraph(state_schema=EnhancedMessagesState)

    # Add the nodes and edges
    workflow.add_node("call_model", lambda state: call_model(state, llm, system_prompt)) # type: ignore
    workflow.add_edge(START, "call_model")
    workflow.add_edge("call_model", END)

    # Set the entry point
    workflow.set_entry_point("call_model")

    # Compile with memory as before
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory) # type: ignore

    return app

def call_model(state: MessagesState, llm: BaseChatModel, system_prompt: SystemMessage) -> dict[str, Any]:
    '''
    Call the model with a structured output and return the response.
    '''
    messages = [system_prompt] + state["messages"]
    raw_response = llm.invoke(messages)

    # Extract content without potential markdown formatting
    content = str(raw_response.content) # type: ignore

    try:
        parsed_output = parser.parse(content)
        # Return the structured output as a dictionary
        return {"messages": state["messages"] + [AIMessage(content=content)], 
                "structured_output": parsed_output.to_dict()} # type: ignore
    except Exception as e:
        print(f"Error parsing model output: {e}")
        return {"messages": state["messages"] + [AIMessage(content=content)],
                "structured_output": {"likelihood": "Error parsing response", 
                                     "justification": "Error parsing response"}} # type: ignore


def generate(harms: list[str], human_rights: list[list[str]], llm: BaseChatModel) -> dict[str, list[list[str]]]:
    '''
    Generates likelihood levels and justifications for the given harms within the context of the corresponding human right using the provided `llm`.
    The function returns a dictionary with likelihood levels and justifications, where each level corresponds to the likelihood of the respective harm.
    The shape of the output is the same as the shape of the input human rights.
   
    Args:
        harms (list[str]): A list of harms to consider.
        human_rights (list[list[str]]): A list of lists human rights for every harm.
        llm (BaseChatModel): The language model to use for generating the likelihood levels and justifications.
    Returns:
        dict[str, list[list[str]]]: A dictionary with two keys: 'likelihood_levels' and 'justifications'.
                                     Each key maps to a list of lists containing the respective levels and justifications.
    '''
    # Define the system prompt
    system_prompt = SystemMessage(content=f"""
    You are an expert in AI ethics and you are asked to evaluate the likelihood of certain harm happening within the context of a certain human right (as defined by the European Convention on Human Rights).
    You are also asked to provide justifications for the likelihood levels.

    Likelihood refers to the level of estimated frequency or degree of possibility of this particular impact occuring.
    Assign the likelihood on a scale from 0.0 to 1.0, where 0.0 means not likely occuring impact, 0.2 means minor likelihood of impact, 0.6 means medium likelihood of impact, 0.8 means high likelihood of impact and 1.0 means certainly occuring impact.
    Provide a brief justification explaining why you assigned that likelihood level.

    Your response MUST be formatted as a valid JSON object that follows this exact structure:
    {parser.get_format_instructions()}

    Do not include any explanations, markdown formatting or additional text outside the JSON.""")

    # Initialize the workflow with the LLM and system prompt
    app = init_workflow(llm, system_prompt)

    # Initialize empty lists to store the likelihood levels and justifications
    all_likelihood_levels: list[list[str]] = []
    all_justifications: list[list[str]] = []

    for i, harm in enumerate(harms):
        # Create a unique thread ID based on the harm to maintain context
        thread_id = f"likelihood_generation_{harm}".replace(" ", "_")

        # Initialize lists to store likelihood levels and their justifications for this harm
        harm_likelihood_levels: list[str] = []
        harm_justifications: list[str] = []

        # Iterate over human rights for this harm
        for human_right in human_rights[i]:
            # Define the user message for the current human right and harm
            user_message = HumanMessage(content=f"Please assess the likelihood of the following harm occuring: {harm} in the context of the following (ECHR) human right: {human_right}.")
        
            # Invoke the LLM with the user message
            response = app.invoke(
                {"messages": [user_message]},
                config={"configurable": {"thread_id": thread_id}},
            )
        
            structured_response = response["structured_output"]

            # Add the severity level and justication to their respective list
            harm_likelihood_levels.append(structured_response['likelihood'])
            harm_justifications.append(structured_response['justification'])
        
        # Add the list of likelihood levels and justifications for this harm to their respective lists
        all_likelihood_levels.append(harm_likelihood_levels)
        all_justifications.append(harm_justifications)
        
    return {"likelihood_levels": all_likelihood_levels, "justifications": all_justifications}