"""
Generate harm descriptions based on vignettes and stakeholders using LLMs.
This module provides functionality to analyze given vignettes (scenarios) and generate 
descriptions of potential harms that specific stakeholders might face due to problematic 
AI behavior. The module uses LangGraph for workflow management and LangChain for 
language model interactions.
Functions:
    init_workflow(llm): Initialize a workflow graph with a language model.
    call_model(state, llm): Process messages through the language model.
    generate(scenario, vignettes, stakeholder, llm): Generate harm descriptions for vignettes 
        from a stakeholder's perspective.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain.schema import HumanMessage, SystemMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

memory = MemorySaver()


def init_workflow(llm: BaseChatModel, system_message: SystemMessage) -> CompiledStateGraph:
    '''
    Initialize the workflow with the state schema and nodes.
    Compile the StateGraph with memory.
    '''
    # Create the workflow
    workflow = StateGraph(state_schema=MessagesState)

    # Add the nodes and edges
    workflow.add_node("call_model", lambda state: call_model(state, llm, system_message)) # type: ignore
    workflow.add_edge(START, "call_model")
    workflow.add_edge("call_model", END)

    # Set the entry point
    workflow.set_entry_point("call_model")

    # Compile with memory as before
    app = workflow.compile(checkpointer=memory) # type: ignore

    return app


def call_model(state: MessagesState, llm: BaseChatModel, system_message: SystemMessage) -> dict[str, BaseMessage]:
    '''
    Call the model and return the response.
    '''
    messages = [system_message] + state["messages"]
    response = llm.invoke(messages)
    return {'messages': response}
    

def generate(scenario: str, vignettes: list[str], stakeholder: str, llm: BaseChatModel) -> list[str]:
    """
    Generate descriptions of harms stakeholders face due to problematic AI behavior.
    This function processes a list of vignettes and generates descriptions of potential harms
    that a specified stakeholder might experience in each scenario. The responses are
    formulated in second-person perspective.
    Args:
        scenario (str): A plain text description of the scenario in which the harms are analyzed.
        vignettes (list[str]): A list of vignette descriptions to analyze for potential harms.
        stakeholder (str): The type of stakeholder experiencing the harm (e.g., "user", "developer").
        llm (BaseChatModel): The language model to use for generating harm descriptions.
    Returns:
        list[str]: A list of generated harm descriptions for each vignette, written from 
                   the stakeholder's perspective.
    """
    system_message = SystemMessage(
        content="You are an expert in analyzing potential harms that stakeholders face due to problematic AI behavior. "
                "You will be given vignettes (scenarios), a stakeholder and a scenario. "
                "Your task is to summarize the vignette and specify the harm the stakeholder faces due to the problematic AI behavior. "
                f"The answer should be specific to the following scenario: {scenario}"
                "Formulate your answer in second-person perspective: 'Imagine you are a [stakeholder], ...'. "
                "Use a few sentences at most and answer in a single paragraph."
    )

    app = init_workflow(llm, system_message)
    
    # Create a unique thread ID based on the scenario and stakeholder to maintain context
    thread_id = f"harm_generation_{vignettes}_{stakeholder}".replace(" ", "_")

    harms: list[str] = list()

    for vignette in vignettes:
        current_message = HumanMessage(content=f"Summarize the vignette {vignette} and specify the harm the stakeholder {stakeholder} faces due to the problematic AI behavior.")
    
        # Invoke the model with the message
        response = app.invoke(
            {"messages": [current_message]},
            config={"configurable": {"thread_id": thread_id}},
        )

        harms.append(response['messages'][-1].content)

    return harms