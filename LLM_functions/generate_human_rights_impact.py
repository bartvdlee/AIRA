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


from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain.schema import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from typing import Optional

# local imports
from data_store import DataManager


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
    

def generate(harms: list[str], stakeholder: str, llm: BaseChatModel | CompiledStateGraph, dm: Optional[DataManager] = None) -> list[list[str]]:
    '''
    Generates a list of human rights affected by the given harms for the specified stakeholder using the provided `llm`.
    The function returns a list of lists, where each inner list contains the affected human rights for each harm.
    The human rights are generated based on the European Convention on Human Rights.
    The function works for BaseChatModel and CompiledStateGraph (RAG) types.
    
    Args:
        harms (list[str]): A list of harms to consider.
        stakeholder (str): The stakeholder for whom to generate the human rights.
        llm (BaseChatModel | CompiledStateGraph): The language model to use for generating the human rights.
        dm (Optional[DataManager]): An optional DataManager instance for saving context when using RAG.
    Returns:
        list[list[str]]: A list of lists, where each inner list contains the affected human rights for each harm.
    '''
    # Create a unique thread ID based on the harms and stakeholder to maintain context
    thread_id = f"human_rights_generation_{harms}_{stakeholder}".replace(" ", "_")

    # Initialize the workflow only for BaseChatModel
    if isinstance(llm, BaseChatModel):
        app = init_workflow(llm)

        human_rights: list[list[str]] = list()

        for harm in harms:
            current_message = HumanMessage(content=f"""What human rights (as defined by the European Convention on Human Rights) of the
                                        stakeholder '{stakeholder}' are affected by this harm: {harm}.
                                        Refrain from mentioning rights and freedoms of other stakeholders, but only focus on the
                                        rights and freedoms of {stakeholder}. Respond by giving a comma-seperated list of the affected rights for the harm.
                                        Provide the name of the right and the article number. Use the following format: Article [article number] ([article name]).
                                        Also mention violated protocols, if applicable.""")
                                        # Use the following format: Article P[protocol number]-[article number] ([protocol name]). 
                                        # Do not provide any additional text or explanation.""")
        
            # Invoke the model with the message
            response = app.invoke(
                {"messages": [current_message]},
                config={"configurable": {"thread_id": thread_id}},
            )

            human_rights.append([item.strip() if item.strip().endswith(')') else item.strip() + ')' for item in str(response['messages'][-1].content).split('),')])

    else:
        app = llm

        human_rights: list[list[str]] = list()
        answer_context: list[str] = list()

        for harm in harms:
            current_message = f"""What human rights (as defined by the European Convention on Human Rights) of the
                                stakeholder '{stakeholder}' are affected by this harm: {harm}.
                                Refrain from mentioning rights and freedoms of other stakeholders, but only focus on the
                                rights and freedoms of {stakeholder}. Respond by giving a comma-seperated list of the affected rights for the harm.
                                Provide the name of the right and the article number. Use the following format: Article [article number] ([article name]).
                                Also mention violated protocols, if applicable. Use the following format: Article P[protocol number]-[article number] ([protocol name]). 
                                Do not provide any additional text or explanation."""
        
            # Invoke the model with the message
            response = app.invoke(
                {"question": current_message},
                # config={"configurable": {"thread_id": thread_id}},
            )

            human_rights.append([item.strip() if not item.endswith('.') else item.strip()[:-1] for item in str(response['answer']).split(',')])
            answer_context.append(str(response['context']))
        
        # Save the context to a json file
        if isinstance(dm, DataManager):
            dm.save_data(f"human_rights_context_{stakeholder}".replace(' ', '_'), answer_context)
        else:
            print("DataManager instance not provided. Context not saved.")

    return human_rights