"""
Generate vignettes and vignettes of harm for AI scenarios from various stakeholder perspectives.
This module provides functionality to generate structured narratives about how different stakeholders 
might experience problematic AI behaviors in specified scenarios. It uses LangChain and LangGraph
to create a structured workflow that:
1. Takes in scenario descriptions, stakeholder information, and problematic behaviors
2. Uses a language model to generate personalized vignettes from the stakeholder's perspective
3. Parses and structures the responses into a consistent format
The module employs a state graph approach to manage the generation workflow and ensures
outputs follow a structured format using Pydantic models.
Classes:
    VignetteGeneration: Pydantic model defining the structure for vignette outputs
    EnhancedMessagesState: Extended state class for the LangGraph workflow
Functions:
    init_workflow: Initialize the generation workflow with model and system prompt
    call_model: Execute the model call with structured output parsing
    generate: Main function to generate vignettes for multiple problematic behaviors
Dependencies:
    - langchain
    - langgraph
    - pydantic
Example:
    ```python
    llm = ChatOpenAI(model="gpt-4o")
    scenario = "An AI content moderation system for a social media platform"
    stakeholder = "A regular user who posts content about political activism"
    problematic_behaviors = [
        "The AI system incorrectly flags legitimate political content as violating guidelines",
        "The AI system allows harmful content while removing educational content"
    ]
    vignettes = generate(scenario, stakeholder, problematic_behaviors, llm)
    ```
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
class VignetteGeneration(BaseModel):
    """Generation of vignette and vignette of specified harm for the stakeholder."""

    vignette: str = Field(description="The vignette (how stakeholder in the scenario may experience the behaviour) of the problematic behaviour. Formulate your answer in second-person perspective: 'Imagine you are a [stakeholder], you may experience [harm] because ...'")
    vignette_of_harm: str = Field(description="The vignette of specified harms (how stakeholder in the scenario may experience the problematic AI behaviour) of the problematic behaviour. Formulate your answer in second-person perspective: 'Imagine you are a [stakeholder], ...'")

    def to_dict(self):
        return {
            "vignette": self.vignette,
            "vignette_of_harm": self.vignette_of_harm
        }

class EnhancedMessagesState(MessagesState):
    structured_output: Optional[dict[str, Any]]

# Create parser
parser = PydanticOutputParser(pydantic_object=VignetteGeneration)

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
                "structured_output": {"vignette": "Error parsing response", 
                                     "vignette_of_harm": "Error parsing response"}} # type: ignore


def generate(scenario: str, stakeholder: str, problematic_behaviours: list[str], llm: BaseChatModel) -> list[dict[str, str]]:
    '''
    Generate vignettes and vignettes of harms for a given scenario, stakeholder, and problematic behaviors
    using a language model with conversation history tracking.
    
    Args:
        scenario: Description of the scenario
        stakeholder: The stakeholder perspective to consider
        problematic_behaviours: List of problematic AI behaviors to analyze
        llm: Language model to use for generation
        
    Returns:
        List with dictionaries of vignettes and vignettes of harm for every problematic behaviour
    '''

    # Define the system prompt
    system_prompt = SystemMessage(content=f"""
    You are an expert in creating vignettes about AI harms and experiences.
    Analyze the following scenario and stakeholder.
    You will get questions about formulating vignettes and vignettes of specified harms for different problematic behaviours.
    
    Your response MUST be formatted as a valid JSON object that follows this exact structure:
    {parser.get_format_instructions()}
    
    Do not include any explanations, markdown formatting or additional text outside the JSON.                        
    
    Scenario: {scenario}
    Stakeholder: {stakeholder}
    """)

    app = init_workflow(llm, system_prompt)
    
    # Create a unique thread ID based on the scenario and stakeholder to maintain context
    thread_id = f"vignette_generation_{scenario}_{stakeholder}".replace(" ", "_")

    stakeholder_vignettes: list[dict[str, str]] = []

    for problematic_behaviour in problematic_behaviours:
        # Add the current problematic behavior prompt
        current_message = HumanMessage(content=f"Generate a vignette and vignette of harms for the problematic behavior: {problematic_behaviour}")
        
        # Invoke the model with the message
        response = app.invoke(
            {"messages": [current_message]},
            config={"configurable": {"thread_id": thread_id}},
        )
        
        # The structured_output field contains the parsed result
        if "structured_output" in response:
            stakeholder_vignettes.append(response["structured_output"])
        else:
            # Fallback in case structured output isn't available
            stakeholder_vignettes.append({
                "vignette": "Failed to generate vignette",
                "vignette_of_harm": "Failed to generate vignette of harms"
            })

    return stakeholder_vignettes