"""
Generates severity levels for the given harms within the context of the specified human rights using the provided language model.
This function evaluates each harm in the context of its associated human rights, generating a severity assessment
for each combination. The language model acts as an AI ethics expert to determine the severity level.
    human_rights (list[list[str]]): A nested list where each inner list contains human rights associated with 
                                   the corresponding harm at the same index in the harms list.
    llm (BaseChatModel): The language model to use for generating the severity assessments.
    list[list[str]]: A nested list of severity levels matching the structure of the input human_rights list.
                     Each severity level is one of: 'low', 'medium', 'high', or 'very high'.
Example:
    ```
    harms = ["Privacy breach", "Discrimination"]
    human_rights = [["Right to privacy"], ["Right to equality", "Right to non-discrimination"]]
    model = ChatOpenAI()
    severity_levels = generate(harms, human_rights, model)
    # Could return: [["high"], ["very high", "high"]]
    ```
"""


from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel


def generate(harms: list[str], human_rights: list[list[str]], llm: BaseChatModel) -> list[list[str]]:
    '''
    Generates severity levels for the given harms within the context of the specified human rights using the provided `llm`.
    The function returns a list of lists of severity levels, where each level corresponds to the severity of the respective harm.
    The shape of the output is the same as the shape of the input human rights.
    
    Args:
        harms (list[str]): A list of harms to consider.
        human_rights (list[list[str]]): A list of lists of human rights for every harm.
        llm (BaseChatModel): The language model to use for generating the severity levels.
    Returns:
        list[list[str]]: A list of lists with severity levels for each harm, where each level is one of 'low', 'medium', 'high', or 'very high'.
    '''
    # Define the system message for the LLM
    system_message = SystemMessage(
        content="You are an expert in AI ethics and you are asked to evaluate the severity levels of certain harms within the context of certain human right. "
                "Please evaluate the severity level for the given harm within the context of the given human right (enshrined in the universal declaration of human rights)."
                "Assign the level of severity on a scale of low, medium, high or very high."
                "Respond only with the severity level, without any additional text or explanation."
    )
    
    # Initialize an empty list to store the severity levels
    severity_levels: list[list[str]] = []
    # Iterate over the harms and human rights
    for i, harm in enumerate(harms):
        # Initialize a list to store severity levels for this harm
        harm_severity_levels: list[str] = []
        # Iterate over human rights for this harm
        for human_right in human_rights[i]:
            # Define the user message with the scenario
            user_message = HumanMessage(
                content=f"Please assess the severity for the following harm: {harm} in the context of the following (ECHR) human right: {human_right}. "
            )
            
            # Invoke the LLM with the system and user messages
            response = llm.invoke([system_message, user_message])
            
            # Add the severity level to the list
            harm_severity_levels.append(str(response.content)) # type: ignore
        
        # Add the list of severity levels for this harm to the main list
        severity_levels.append(harm_severity_levels)
    
    return severity_levels