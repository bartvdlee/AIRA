import pandas as pd
# from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
import os

from LLM import init_llm, get_available_models
from modify_report import choose_report_to_modify
from data_store import DataManager


def analyse_harms(report_name: str, llm_name: str, use_responses_api: bool = False) -> None:
    '''
    Analyse harms for a given report using the specified language model.
    Args:
        report_name (str): The name of the report to analyse harms for.
        llm (BaseChatModel): The language model to use for the analysis.
    '''
    # init the LLM
    if not use_responses_api:
        llm = init_llm(llm_name)
    else:
        llm = init_llm(llm_name, use_responses_api=True)

    # Load the report info
    info: pd.DataFrame = pd.read_csv(f"reports/{report_name}/info.csv", index_col=0)  # type: ignore
    scenario = str(info.loc['scenario'].iloc[0])  # type: ignore

    # Initialise the DataManager
    dm = DataManager(report_name)

    # Load stakeholder data
    stakeholders = dm.load_data('stakeholders')

    # Create a variable for all the stakeholders
    all_stakeholders = stakeholders['direct_stakeholders'] + stakeholders['indirect_stakeholders']

    # Load harms data
    harms = dm.load_data('harms')

    prompt = ChatPromptTemplate.from_messages(  # type: ignore
        [
            ("system", "You are an expert in analysing different harms for AI systems in a certain scenario for a certain stakeholder. "
             "You will be given possible harms and you will analyse whether it is a meaningful harm or not. "
             "A harm is meaningful if all these three conditions are satisfied: (i) the connection between action and consequences is plausible or logical; (ii) the consequences have actual harmful effects; (iii) the harm is affecting the target stakeholder"
             "Only respond with 'meaningful' or 'non-meaningful'. Do not provide any additional information or explanations. "
             "scenario: {scenario}"),
            ("human", "{query}"),
        ]).partial(scenario=scenario)
    
    # Create a list to store the results
    results = []

    for stakeholder in all_stakeholders:
        print(f"\nAnalysing harms for stakeholder: {stakeholder}")

        for harm in harms[stakeholder]:
            # Invoke the model with the message
            message = prompt.invoke({"query": f"Is this harm meaningful for the stakeholder '{stakeholder}': {harm}?"}) # type: ignore
            response = llm.invoke(message)

            if use_responses_api:
                response = response.content[0] # type: ignore
                response = response['text'] # type: ignore
            else:
                response = response.content # type: ignore

            # Process the response
            if str(response).lower() == 'meaningful':  # type: ignore
                results.append({'stakeholder': stakeholder, 'harm': harm, 'meaningful': True})  # type: ignore
            elif str(response).lower() == 'non-meaningful':  # type: ignore
                results.append({'stakeholder': stakeholder, 'harm': harm, 'meaningful': False})  # type: ignore
            else:
                results.append({'stakeholder': stakeholder, 'harm': harm, 'meaningful': None}) # type: ignore
    
    # Save the results to a CSV file
    df_results = pd.DataFrame(results)

    # # Ensure the 'harm_analysis' directory exists before saving
    # os.makedirs(f'reports/{report_name}/harm_analysis', exist_ok=True)

    path = f"reports/{report_name}/harm_analysis/{llm_name.split('/')[-1].replace(' ', '_')}.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_results.to_csv(path, index=False)


if __name__ == '__main__':
    # Ask the user to choose a report
    report_name = choose_report_to_modify()

    # Ask the user to choose a language model
    llm_name = get_available_models()


    # Analyse harms for the chosen report
    analyse_harms(report_name, llm_name, use_responses_api=True)

   