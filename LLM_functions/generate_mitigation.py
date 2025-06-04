"""
This module provides functionality for generating mitigation measures for various harms and human rights
using a language model. It utilizes a workflow-based approach to maintain state and context during
the generation process.
The module contains functions to initialize a workflow, call a language model, and generate
mitigation measures for specific stakeholders, harms, and impacted human rights.
    Invokes the language model with the current state's messages and returns the response.
        state (MessagesState): The current state containing messages to be sent to the model.
        llm (BaseChatModel): The language model to invoke.
        dict: A dictionary with the 'messages' key containing the model's response.
"""


from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain.schema import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph



memory = MemorySaver()


def init_workflow(llm: BaseChatModel) -> CompiledStateGraph:
    '''
    Initialize the workflow with the state schema and nodes.
    '''
    # Create the workflow
    workflow = StateGraph(state_schema=MessagesState)

    # Add the nodes and edges
    workflow.add_node("call_model", lambda state: call_model(state, llm)) # type: ignore
    workflow.add_edge(START, "call_model")
    workflow.add_edge("call_model", END)

    # Set the entry point
    workflow.set_entry_point("call_model")

    # Compile with memory
    app = workflow.compile(checkpointer=memory) # type: ignore

    return app


def call_model(state: MessagesState, llm: BaseChatModel):
    response = llm.invoke(state["messages"])
    return {'messages': response}


def generate(stakeholder: str, harms: list[str], human_rights: list[list[str]], llm: BaseChatModel) -> list[list[str]]:
    """
    Generates mitigation measures for every provided human right within the given harms and for the given stakeholder.
    The function uses the provided language model to generate the list of lists with mitigation measures.
    
    Args:
        stakeholder (str): The stakeholder for whom to generate the mitigation measures.
        harms (list[str]): A list of harms to consider.
        human_rights (list[list[str]]): A list of lists, where each inner list contains the impacted human rights for each harm.
        llm (BaseChatModel): The language model to use for generating the mitigation measures.
    Returns:
        list[list[str]]: A list of lists, where each inner list contains the mitigation measures for each harm.
    """
    app = init_workflow(llm)
    
    # Create a unique thread ID based on the scenario and stakeholder to maintain context, shorten it a bit to prevent memory buildup
    thread_id = f"mitigation_generation_{stakeholder}_{harms[:20]}".replace(" ", "_")

    mitigation_measures: list[list[str]] = list()

    for harm, human_rights_list in zip(harms, human_rights):
        mitigations_for_current_harm: list[str] = list()

        for human_right in human_rights_list:
            current_message = HumanMessage(content=f"""What possible mitigation measure can be taken to prevent or reduce the impact of the following harm: {harm} 
                                                    for the stakeholder '{stakeholder}' in the context of the following (ECHR) human right: {human_right}.
                                                    Respond in at most a couple of sentences""")
            
            # Invoke the model with the message
            response = app.invoke(
                {"messages": [current_message]},
                config={"configurable": {"thread_id": thread_id}},
            )

            mitigations_for_current_harm.append(str(response['messages'][-1].content))

        mitigation_measures.append(mitigations_for_current_harm)

    return mitigation_measures