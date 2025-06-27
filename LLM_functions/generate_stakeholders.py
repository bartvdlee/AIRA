"""
Stakeholder Generation Module
This module provides functionality to generate comprehensive lists of direct and indirect stakeholders
for AI system scenarios. It uses a language model to identify relevant stakeholders based on a given 
scenario description and user-provided initial stakeholders.
The module defines:
- A Pydantic model for structured stakeholder generation output
- A function to generate categorized stakeholder lists
This is part of the AIRA (Automated Impact on Rights Assessment) pipeline, helping
to ensure thorough stakeholder identification for AI system impact analysis for the EU AI act.
"""


from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel


class stakeholder_generation(BaseModel):
        """Generation of lists of direct and indirect stakeholders."""

        direct_stakeholders: str = Field(description="Comma-separated list of direct stakeholders (stakeholders who directly interact with or are immediately affected by the AI system)")
        indirect_stakeholders: str = Field(description="Comma-separated list of indirect stakeholders (stakeholders who are indirectly affected by the AI system, like people associated with direct stakeholders or larger community groups)")


def generate(scenario: str, user_specified_stakeholders: list[str], llm: BaseChatModel) -> dict[str, list[str]]:
    '''
    Generates a list of stakeholders for the given scenario using the provided `llm` in the given `scenario`.
    It uses the `user-specified stakeholders` as inspiration.
    The function returns a dictionary with as keys the category of stakeholders (direct or indirect) and as values the stakeholders.

    Args:
        scenario (str): The scenario for which to generate stakeholders.
        user_specified_stakeholders (list[str]): A list of user-specified stakeholders to use as inspiration.
        llm (BaseChatModel): The language model to use for generating the stakeholders.
    
    Returns:
        dict[str, list[str]]: A dictionary with as keys the categories of stakeholders (direct or indirect) and as values the stakeholders.
    '''
    structured_llm = llm.with_structured_output(stakeholder_generation) # type: ignore

    response_data = structured_llm.invoke(f"Come up with a list of stakeholders for the following scenario: {scenario}. Take inspiration (and include relevant stakeholders) from this list: {str(user_specified_stakeholders)[1:-1]}" # type: ignore
                                          "Give at most 10 direct stakeholders and 10 indirect stakeholders and at least 5 direct stakeholders and 5 indirect stakeholders.")
    structured_response = stakeholder_generation.model_validate(response_data)

    direct_stakeholders = [str(item).strip() for item in structured_response.direct_stakeholders.split(',')]
    indirect_stakeholders = [str(item).strip() for item in structured_response.indirect_stakeholders.split(',')]

    # Create a dictionary with the categories and stakeholders
    stakeholders_dict = {
        'direct_stakeholders': direct_stakeholders,
        'indirect_stakeholders': indirect_stakeholders
    }

    return stakeholders_dict