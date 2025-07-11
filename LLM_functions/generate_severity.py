"""
Generates severity levels for the given harms within the context of the specified human rights using the provided language model.
This function evaluates each harm in the context of its associated human rights, generating a severity assessment
for each combination. The language model acts as an AI ethics expert to determine the severity level.
    human_rights (list[list[str]]): A nested list where each inner list contains human rights associated with 
                                   the corresponding harm at the same index in the harms list.
    llm (BaseChatModel): The language model to use for generating the severity assessments.
    list[list[str]]: A nested list of severity levels matching the structure of the input human_rights list.
                     Each severity level is one on a scale of 0.0 to 1.0.
"""
from typing import Optional, Any
from pydantic import BaseModel, Field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.output_parsers import PydanticOutputParser
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver


# Define a Pydantic model for structured output
class SeverityGeneration(BaseModel):
        """Generation of severity levels for harms in the context of human rights and justification."""
        severity_level: str = Field(description="Severity of the harm within the context of the given human right")
        justification: str = Field(description="Justification for the choice of the severity level")

        def to_dict(self):
            return {
                "severity_level": self.severity_level,
                "justification": self.justification
            }


class EnhancedMessagesState(MessagesState):
    structured_output: Optional[dict[str, Any]]


# Create parser
parser = PydanticOutputParser(pydantic_object=SeverityGeneration)


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

    # Compile with memory
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
                "structured_output": {"severity_level": "Error parsing response", 
                                     "justification": "Error parsing response"}} # type: ignore


def generate(harms: list[str], human_rights: list[list[str]], llm: BaseChatModel) -> dict[str, list[list[str]]]:
    '''
    Generates severity levels for the given harms within the context of the specified human rights using the provided `llm`.
    The function returns a list of lists of severity levels, where each level corresponds to the severity of the respective harm.
    The shape of the lists within the dict is the same as the shape of the input human rights.
    The output dict contains two keys:
        - "severity_levels": A list of lists of severity levels for each harm, where each level is on a scale from 0.0 to 1.0.
        - "justifications": A list of lists of justifications for the severity levels, explaining the reasoning behind each level.
    
    Args:
        harms (list[str]): A list of harms to consider.
        human_rights (list[list[str]]): A list of lists of human rights for every harm.
        llm (BaseChatModel): The language model to use for generating the severity levels.
    Returns:
        dict[str, list[list[str]]]: A dict with str and list of lists with severity levels for each harm, where each level is on a scale from 0.0 to 1.0.
    '''
    # Define the system message for the LLM
    system_message = SystemMessage(
        content="You are an expert in AI ethics and you are asked to evaluate the severity levels of certain harms within the context of certain human right. "
                "Please evaluate the severity level for the given harm within the context of the given human right (as enshrined in the European Convention on Human Rights)."
                "Assign the level of severity on a scale from 0.0 to 1.0, where 0.0 means neglible impact, 0.2 means minor impact, 0.4 means moderate impact, 0.6 means significant discomfort, 0.8 means severe impact and 1.0 means life-altering or threatening impact."
                "Generate the level of severity of impact of the harm based on this scale and justify your choice of level."
                "Your response MUST be formatted as a valid JSON object that follows this exact structure:"
                f"{parser.get_format_instructions()}"
                "Do not include any explanations, markdown formatting or additional text outside the JSON." 
    )

    # Initialize the workflow with the LLM and system message
    app = init_workflow(llm, system_message)
    
    # Initialize empty lists to store the severity levels and their justifications
    severity_levels: list[list[str]] = []
    justifications: list[list[str]] = []

    # Iterate over the harms and human rights
    for i, harm in enumerate(harms):
        # Create a unique thread ID based on the harm to maintain context
        thread_id = f"severity_generation_{harm}".replace(" ", "_")

        # Initialize lists to store severity levels and their justifications for this harm
        harm_severity_levels: list[str] = []
        harm_justifications: list[str] = []

        # Iterate over human rights for this harm
        for human_right in human_rights[i]:
            # Define the user message with the scenario
            user_message = HumanMessage(
                content=f"Please assess the severity for the following harm: {harm} in the context of the following (ECHR) human right: {human_right}."
            )
            
            # Invoke the LLM with the user message
            response = app.invoke(
                {"messages": [user_message]},
                config={"configurable": {"thread_id": thread_id}},
            )
            
            structured_response = response["structured_output"]

            # Add the severity level and justication to their respective list
            harm_severity_levels.append(structured_response['severity_level'])
            harm_justifications.append(structured_response['justification'])
        
        # Add the list of severity levels and justifications for this harm to their respective lists
        severity_levels.append(harm_severity_levels)
        justifications.append(harm_justifications)
    
    return {"severity_levels": severity_levels, "justifications": justifications}