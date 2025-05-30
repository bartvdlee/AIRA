"""
Generate vignettes for AI scenarios from various stakeholder perspectives.
This module provides functionality to generate structured narratives about how different stakeholders 
might experience problematic AI behaviors in specified scenarios. It uses LangChain and LangGraph
to create a structured workflow that:
1. Takes in scenario descriptions, stakeholder information, and problematic behaviors
2. Uses a language model to generate personalized vignettes from the stakeholder's perspective
3. Parses and structures the responses into a consistent format
The module employs a state graph approach to manage the generation workflow and ensures
outputs follow a structured format using Pydantic models.
Functions:
    init_workflow: Initialize the generation workflow
    call_model: Execute the model
    generate: Main function to generate vignettes for multiple problematic behaviors
Dependencies:
    - langchain
    - langgraph
"""
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.schema import HumanMessage
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()


def init_workflow(llm: BaseChatModel) -> CompiledStateGraph:
    '''
    Initialize the workflow with the state schema and nodes.
    Compile the StateGraph with memory.
    '''
    # Create the workflow
    workflow = StateGraph(state_schema=MessagesState)

    # Add the nodes and edges
    workflow.add_node("call_model", lambda state: call_model(state, llm)) # type: ignore
    workflow.add_edge(START, "call_model")
    workflow.add_edge("call_model", END)

    # Set the entry point
    workflow.set_entry_point("call_model")

    # Compile with memory as before
    app = workflow.compile(checkpointer=memory) # type: ignore

    return app


def call_model(state: MessagesState, llm: BaseChatModel):
    '''
    Call the model and return the response.
    '''
    response = llm.invoke(state["messages"])
    return {'messages': response}


def generate(scenario: str, stakeholder: str, problematic_behaviours: list[str], llm: BaseChatModel) -> list[str]:
    '''
    Generate vignettes for a given scenario, stakeholder, and problematic behaviors
    using a language model with conversation history tracking.
    
    Args:
        scenario: Description of the scenario
        stakeholder: The stakeholder perspective to consider
        problematic_behaviours: List of problematic AI behaviors to analyze
        llm: Language model to use for generation
        
    Returns:
        List with strings of generated vignettes
    '''
    app = init_workflow(llm)
    
    # Create a unique thread ID based on the scenario and stakeholder to maintain context
    thread_id = f"vignette_generation_{scenario}_{stakeholder}".replace(" ", "_")

    vignettes: list[str] = []

    for problematic_behaviour in problematic_behaviours:
        current_message = HumanMessage(content=f"Narrate how stakeholder '{stakeholder}' in the scenario '{scenario}' may experience {problematic_behaviour}. \
                                                    Formulate your answer in second-person perspective: 'Imagine you are a [stakeholder], you may experience [harm] because...'")
    
        # Invoke the model with the message
        response = app.invoke(
            {"messages": [current_message]},
            config={"configurable": {"thread_id": thread_id}},
        )

        vignettes.append(response['messages'][-1].content)

    return vignettes