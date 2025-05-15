"""
Improves a scenario for an AI Fundamental Rights Impact Assessment by using a language model.
This function takes a scenario text and passes it to the provided language model (LLM),
instructing the model to act as an AI ethics expert. The model then generates a more
detailed and comprehensive version of the scenario, limited to a couple of sentences.
    scenario (str): The original scenario text to be improved. This should be a brief
        description of a situation for which an AI Fundamental Rights Impact Assessment
        is being conducted.
    llm (BaseChatModel): The language model instance that will generate the improved
        scenario. Must be an instance of a class derived from BaseChatModel.
Returns:
    str: The improved scenario text, with more detail and comprehensive description
        than the original, but still concise (a couple of sentences).
Note:
    The quality of the improvement depends on the capabilities of the provided
    language model. The function instructs the model to keep responses brief and
    focused on the scenario itself without additional explanations.
"""


from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel


def improve(scenario: str, llm: BaseChatModel) -> str:
    '''
    Improves the given scenario using the provided `llm`.
    The function returns the improved scenario.

    Args:
        scenario (str): The scenario to improve.
        llm (BaseChatModel): The language model to use for improving the scenario.
    Outputs:
        str: The improved scenario.
    '''
    # Define the system message for the LLM
    system_message = SystemMessage(
        content="You are an expert in AI ethics and you are asked to improve a given scenario for a AI Fundamental Rights Impact Assessment. "
                "Please provide a more detailed and comprehensive version of the scenario. Respond only with the improved scenario, without any additional text or explanation."
                "Use at most a couple of sentences."
    )
    
    # Define the user message with the scenario
    user_message = HumanMessage(
        content=f"Please improve the following scenario: {scenario}"
    )
    
    # Invoke the LLM with the system and user messages
    response = llm.invoke([system_message, user_message])
    
    # Extract and return the improved scenario from the response
    improved_scenario = str(response.content) # type: ignore
    
    return improved_scenario