"""
Module for generating likelihood and confidence levels for potential human rights harms.
This module provides functionality to evaluate the likelihood of specific harms 
occurring within the context of human rights using a language model. It returns
structured assessments with both likelihood levels and confidence scores.
Classes:
    likelihood_generation: Pydantic model for structured output representing 
                          likelihood and confidence levels.
Functions:
    generate: Evaluates likelihood and confidence levels for harms in the context 
             of human rights using a language model.
Example:
    >>> from langchain_core.language_models import ChatOpenAI
    >>> harms = ["Privacy invasion", "Discrimination"]
    >>> human_rights = [
    ...     ["Right to privacy", "Right to freedom"],
    ...     ["Right to equality", "Right to non-discrimination"]
    ... ]
    >>> llm = ChatOpenAI()
    >>> results = generate(harms, human_rights, llm)
    >>> print(results)
    {
        'likelihood': [['Medium', 'Low'], ['High', 'High']],
        'confidence': [['High', 'Medium'], ['High', 'High']]
    }
"""

from typing import Optional, Any
from pydantic import BaseModel, Field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.output_parsers import PydanticOutputParser
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver


class likelihood_generation(BaseModel):
    """A Pydantic model representing structured output for likelihood and confidence levels 
    associated with human rights harms. This model includes fields for likelihood and 
    likelihood: str = Field(description="Comma-separated likelihood levels for the given human rights and harm")"""

    likelihood: str = Field(description="Comma-separated list of likelihood levels for the given human rights within the context of the given harm")
    confidence: str = Field(description="Comma-separated list of confidence levels for the likelihood levels")

    def to_dict(self):
        return {
            "likelihood": self.likelihood,
            "confidence": self.confidence
        }

class EnhancedMessagesState(MessagesState):
    structured_output: Optional[dict[str, Any]]

# Create parser
parser = PydanticOutputParser(pydantic_object=likelihood_generation)

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
                                     "confidence": "Error parsing response"}} # type: ignore


def generate(harms: list[str], human_rights: list[list[str]], llm: BaseChatModel) -> dict[str, list[list[str]]]:
    '''
    Generates likelihood levels for the given harms within the context of the corresponding human right using the provided `llm`.
    The function returns a list of lists of likelihood levels, where each level corresponds to the likelihood of the respective harm.
    The shape of the output is the same as the shape of the input human rights.
   
    Args:
        harms (list[str]): A list of harms to consider.
        human_rights (list[list[str]]): A list of lists human rights for every harm.
        llm (BaseChatModel): The language model to use for generating the likelihood levels.
    Returns:
        dict[str, list[list[str]]]: A dictionary with two keys: 'likelihood' and 'confidence'.
                                     Each key maps to a list of lists containing the respective levels.
    '''

    # Define the system prompt
    system_prompt = SystemMessage(content=f"""
    You are an expert in AI ethics and you are asked to evaluate the likelihood of certain harms happening within the context of certain human rights (as defined by the European Convention on Human Rights).
    You are also asked to evaluate the confidence level of the likelihood levels.

    Please evaluate the likelihood of the given harm happening within the context of the given human rights.
    Assign the likelihood of the harm happening on a scale of Low, Medium, High or Very high.
    Assign the confidence level of the likelihood level on a scale of Low, Medium, High or Very high.

    YOUR RESPONSE MUST INCLUDE BOTH LIKELIHOOD AND CONFIDENCE LEVELS.

    Example valid response:
    {{
    "likelihood": "Low, Medium, High",
    "confidence": "Medium, High, Low"
    }}

    Your response MUST be formatted as a valid JSON object that follows this exact structure:
    {parser.get_format_instructions()}

    Do not include any explanations, markdown formatting or additional text outside the JSON.  
""")

    app = init_workflow(llm, system_prompt)

    # Create a unique thread ID based on the human rights and harm to maintain context
    thread_id = f"vignette_generation_{human_rights}_{harms}".replace(" ", "_")

    all_likelihood_levels: list[list[str]] = []
    all_confidence_levels: list[list[str]] = []

    for i, harm in enumerate(harms):
         # Add the current likelihood level prompt
        current_message = HumanMessage(content=f"Generate the likelihood and confidence for the following human rights in the context of the following harm. human rights: {human_rights[i]} harm: {harm}.")
        
        # Invoke the model with the message
        response = app.invoke(
            {"messages": [current_message]},
            config={"configurable": {"thread_id": thread_id}},
        )
        
        structured_response = response['structured_output']

        # Extract the likelihood and confidence levels
        likelihood_levels = [str(item).strip() for item in structured_response['likelihood'].split(',')]
        confidence_levels = [str(item).strip() for item in structured_response['confidence'].split(',')]
        
        # Add the likelihood levels and confidence levels to the lists
        all_likelihood_levels.append(likelihood_levels)
        all_confidence_levels.append(confidence_levels)
        
    return {"likelihood": all_likelihood_levels, "confidence": all_confidence_levels}