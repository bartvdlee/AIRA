import os
import pandas as pd
from ast import literal_eval
from tenacity import retry, wait_random_exponential
from langgraph.graph.state import CompiledStateGraph

# local imports
import ask_confirmation
from LLM import full_AFRIA, init_llm, get_available_models, setup_RAG
from LLM_functions import improve_scenario, generate_stakeholders, generate_vignettes, generate_harm, generate_human_rights_impact
from data_store import DataManager


def choose_report_to_modify() -> str:
    '''
    CLI tool to choose a report to modify.
    The function lists all the reports stored in the 'reports' directory and prompts the user to select one by entering its corresponding number.
    The function returns the name of the selected report.

    Returns:
        str: The name of the selected report.
    '''
    print('Below is a list of all the stored reports.')

    # Gets all the reports stored in the reports folder and puts them into a dictionary
    stored_reports = os.listdir('reports')
    reports_dict = {index_no: report for index_no, report in enumerate(stored_reports, start=1)}

    # Prints all the reports from the dictionary with their corresponding keys
    for key, report in reports_dict.items():
        print(f'{key}. {report}')

    # Asks the user to input a report and only continues when the user inputs a valid option
    report_choice = 0
    while report_choice not in reports_dict.keys():
        report_choice = input('\nPlease specify the number of the report you want to modify (e.g. 1): ')

        if report_choice.isdigit():
            try:
                report_choice = int(report_choice)
            except ValueError:
                print('Input is not a number')

            if report_choice not in reports_dict.keys():
                print('The provided number is not a valid option')
        else:
            print('Please input a number')

    report = str(reports_dict[int(report_choice)])

    # Validates the report by checking if the info CSV file exists
    # If the report is valid, return the report name
    # If the report is not valid, rerun the function
    if validate_report(report):
        return report
    else:
        print()
        return choose_report_to_modify()


def validate_report(report_name: str) -> bool:
    '''
    Validates the report by checking if the info CSV file exists, contains the required columns and the columns are not empty.
    The function returns True if the report is valid, otherwise it returns False.

    Args:
        report_name (str): The name of the report to validate.
    Returns:
        bool: True if the report is valid, False otherwise.
    '''
    # Checks if the CSV file exists
    csv_file_path = f"reports/{report_name}/info.csv"

    # If the file does not exist, return False
    if not os.path.exists(csv_file_path):
        print('The report is not valid. Please ensure the info.csv file exists in the report folder.')
        return False

    # Check whether the CSV file contains the required data in the Series index
    required_data: list[str] = ['scenario', 'user_specified_stakeholders', 'problematic_behaviours']
    info = pd.read_csv(csv_file_path, index_col=0) # type: ignore

    if not all(item in info.index for item in required_data): # type: ignore
        print('The report does not contain the required data (scenario, stakeholders, problematic behaviours)')
        return False
    
    # Check whether the columns of the CSV file are not empty
    if not info.notna().all(axis=1).all(): # type: ignore
        print('The report contains empty columns')
        return False

    return True


def modify(report_name: str) -> None:
    '''
    CLI tool to modify an existing report.
    The function does not return any value.
    '''
    dm = DataManager(report_name)
    
    if ask_confirmation.closed('Do you want to use a custom LLM (default: Mistral Small) (y/n)?'):
        llm = init_llm(get_available_models())
        print('Successfully loaded the custom LLM.', end='\n\n')
    else:
        print('Using the default LLM.', end='\n\n')
        llm = init_llm()

    # Asks the user what it wants to modify
    print('What do you want to modify?')
    print()
    print('Report information:')
    print('1. Scenario')
    print('2. User-specified stakeholders')
    print('3. Problematic behaviours')
    print()
    print('Report output (LLM functions):')
    print('4. Run the full AFRIA process')
    print('5. Improve the scenario')
    print('6. Modify generated stakeholders')
    print('7. Modify generated vignettes')
    print('8. Modify generated harms')
    print('9. Generate human rights impacts using RAG')
    print('10. Modify generated mitigation measures')
    print('11. Modify generated severity levels')
    print('12. Modify generated likelihood levels')
    print('13. Modify generated likelihood confidence levels')
    print()

    choice = 0
    while choice not in range(1, 14):
        choice = input('Enter your choice: ')
        if choice.isdigit():
            choice = int(choice)
            if choice not in range(1, 14):
                print('Invalid choice. Please enter a number between 1 and 13.')
        else:
            print('Please input a number')

    if not isinstance(choice, int):
        raise TypeError('Choice is not an integer')

    info: pd.DataFrame = pd.read_csv(f"reports/{report_name}/info.csv", index_col=0) # type: ignore

    # Calls the function to modify the report based on the user's choice
    if choice == 1:
        modify_report_info(report_name, 'scenario')
    elif choice == 2:
        modify_report_info(report_name, 'user_specified_stakeholders', expect_list=True)
    elif choice == 3:
        modify_report_info(report_name, 'problematic_behaviours', expect_list=True)
    elif choice == 4:
        full_AFRIA(report_name, llm, resume=False)

    elif choice == 5:
        scenario = str(info.loc['scenario'].iloc[0]) # type: ignore
        improved_scenario = improve_scenario.improve(scenario, llm)

        checkpoint_data = dm.load_data("improved_scenario")

        print(f'Original scenario: {scenario}')
        print(f"Saved improved scenario: {improved_scenario}")
        print(f"New improved scenario: {checkpoint_data['improved']}")

        # Ask the user if they want to use the new improved scenario
        if ask_confirmation.closed('Do you want to overwrite the saved improved scenario with the new improved scenario (y/n)?'):
            dm.save_data("improved_scenario", {"original": scenario, "improved": improved_scenario})
            print("Successfully updated the improved scenario.")
        else:
            print("Keeping the original improved scenario.")

    elif choice == 6:
        # Modify generated stakeholders

        # Read the report info
        scenario = str(info.loc['scenario'].iloc[0]) # type: ignore
        user_identified_stakeholders = str(info.loc['user_specified_stakeholders'].iloc[0]) # type: ignore

        # Convert the str to a list
        user_identified_stakeholders = literal_eval(user_identified_stakeholders)

        # Generate the stakeholders
        generated_stakeholders = generate_stakeholders.generate(scenario, user_identified_stakeholders, llm)

        # Load the saved stakeholders
        checkpoint_data = dm.load_data("stakeholders")

        print('\nUser-identified stakeholders:')
        print(', '.join(user_identified_stakeholders), end='\n\n')

        print('Saved generated stakeholders:')
        print('Direct stakeholders:')
        print(', '.join(checkpoint_data['direct_stakeholders']))
        print('Indirect stakeholders:')
        print(', '.join(checkpoint_data['indirect_stakeholders']), end='\n\n')

        print('New generated stakeholders:')
        print('Direct stakeholders:')
        print(', '.join(generated_stakeholders['direct_stakeholders']))
        print('Indirect stakeholders:')
        print(', '.join(generated_stakeholders['indirect_stakeholders']), end='\n\n')

        # Ask the user if they want to use the new generated stakeholders
        if ask_confirmation.closed('Do you want to overwrite the saved generated stakeholders with the new generated stakeholders (y/n)?'):
            dm.save_data("stakeholders", generated_stakeholders) # type: ignore
            print("Successfully updated the generated stakeholders.")
        else:
            print("Keeping the original generated stakeholders.")

    elif choice == 7:
        # Modify generated vignettes

        # Read the report info
        scenario = str(info.loc['scenario'].iloc[0]) # type: ignore
        problematic_behaviours = str(info.loc['problematic_behaviours'].iloc[0]) # type: ignore
        problematic_behaviours = literal_eval(problematic_behaviours)

        # Read the saved vignettes
        checkpoint_data = dm.load_data("vignettes")

        # Asks the user to input a stakeholder
        chosen_stakeholder = choose_stakeholder_to_modify(dm)

        # Generate the vignettes
        vignette_list_of_dict = generate_vignettes.generate(scenario, chosen_stakeholder, problematic_behaviours, llm)

        vignettes: list[str] = []
        vignettes_of_harms: list[str] = []
        for dictionary in vignette_list_of_dict:
            vignettes.append(dictionary['vignette'])
            vignettes_of_harms.append(dictionary['vignette_of_harm'])

        # Print the saved and new generated vignettes
        print('\nSaved generated vignettes:')
        for i, vignette in enumerate(checkpoint_data['vignettes'][chosen_stakeholder], start=1):
            print(f"{i}. {vignette}")
        print()

        print('\nNew generated vignettes:')
        for i, vignette in enumerate(vignettes, start=1):
            print(f"{i}. {vignette}")
        print()

        # Ask the user if they want to use the new generated vignettes
        if ask_confirmation.closed('Do you want to overwrite the saved vignettes with the newly generated vignettes (y/n)?'):
            checkpoint_data['vignettes'][chosen_stakeholder] = vignettes
            print("Successfully updated the vignettes.")
        else:
            print("Keeping the original generated vignettes.")

        # Print the saved and new generated vignettes of harms
        print('\nSaved generated vignettes of harms:')
        for i, vignette_of_harm in enumerate(checkpoint_data['vignettes_of_harms'][chosen_stakeholder], start=1):
            print(f"{i}. {vignette_of_harm}")
        print()

        print('\nNew generated vignettes of harms:')
        for i, vignette_of_harm in enumerate(vignettes_of_harms, start=1):
            print(f"{i}. {vignette_of_harm}")
        print()

        # Ask the user if they want to use the new generated vignettes of harms
        if ask_confirmation.closed('Do you want to overwrite the saved vignettes of harms with the newly generated vignettes of harms (y/n)?'):
            checkpoint_data['vignettes_of_harms'][chosen_stakeholder] = vignettes_of_harms
            print("Successfully updated the vignettes of harms.")
        else:
            print("Keeping the original generated vignettes of harms.")

        # Save the modified vignettes
        dm.save_data("vignettes", checkpoint_data)

    elif choice == 8:
        # Modify generated harms

        # Read the saved vignettes
        saved_vignettes: dict[str, list[str]] = dm.load_data("vignettes")['vignettes']
        
        # Asks the user to input a stakeholder
        chosen_stakeholder = choose_stakeholder_to_modify(dm)

        # Read the saved harms
        checkpoint_data = dm.load_data("harms")

        # Generate the harms
        harms: list[str] = generate_harm.generate(saved_vignettes[chosen_stakeholder], chosen_stakeholder, llm)

        # Print the saved and new generated harms        
        print(f'\nSaved generated harms for stakeholder \'{chosen_stakeholder}\':')
        for i, harm in enumerate(checkpoint_data[chosen_stakeholder], start=1):
            print(f"{i}. {harm}")
        print()

        print(f'\nNew generated harms for stakeholder \'{chosen_stakeholder}\':')
        for i, harm in enumerate(harms, start=1):
            print(f"{i}. {harm}")
        print()

        # Ask the user if they want to use the new generated stakeholders
        if ask_confirmation.closed(f'Do you want to overwrite the saved generated harms for stakeholder \'{chosen_stakeholder}\' with the new generated harms (y/n)?'):
            checkpoint_data[chosen_stakeholder] = harms
            dm.save_data("harms", checkpoint_data)
            print("Successfully updated the generated harms.")
        else:
            print("Keeping the original generated harms.")

    elif choice == 9:
        # Human rights impacts generation using RAG

        dm_context = DataManager(report_name)
        dm_context.data_dir = dm_context.report_dir / "data" / "human_rights_context"
        dm_context.data_dir.mkdir(parents=True, exist_ok=True)

        # Read the saved stakeholders
        saved_stakeholders = dm.load_data("stakeholders")
        all_saved_stakeholders = saved_stakeholders['direct_stakeholders'] + saved_stakeholders['indirect_stakeholders']

        # Read the saved harms
        saved_harms = dm.load_data("harms")

        # Setup RAG
        RAG = setup_RAG(llm)

        # retry to get around rate limits
        @retry(wait=wait_random_exponential(multiplier=1, max=60))
        def _gen_with_retry(harms: list[str], stakeholder: str, llm: CompiledStateGraph, dm: DataManager | None = None) -> list[list[str]]:
            """
            Retry function to get around rate limits.
            """
            return generate_human_rights_impact.generate(harms, stakeholder, llm, dm)

        # Generate the human rights impacts using RAG
        human_rights: dict[str, list[list[str]]] = {}
        for stakeholder in all_saved_stakeholders:
            human_rights[stakeholder] = _gen_with_retry(saved_harms[stakeholder], stakeholder, RAG, dm_context)

        # Save the data using the DataManager
        dm.save_data("human_rights", human_rights)
        
        print('Successfully generated human rights impact for the harms using RAG.')


    # elif choice == 10:
    #     modify_generated_mitigation_measures(report_name)
    # elif choice == 11:
    #     modify_generated_severity_levels(report_name)
    # elif choice == 12:
    #     modify_generated_likelihood_levels(report_name)
    # elif choice == 13:
    #     modify_generated_likelihood_confidence_levels(report_name)


def modify_report_info(report_name: str, column_name: str, expect_list: bool = False) -> None:
    '''
    Modifies the report information in the CSV file.
    The function does not return any value.

    Args:
        report_name (str): The name of the report to modify.
        column_name (str): The name of the column to modify.
        expect_list (bool): Whether the column value is expected to be a list or not. Default is False.
    Returns:
        None
    '''
    # Reads the report info
    info: pd.DataFrame = pd.read_csv(f"reports/{report_name}/info.csv", index_col=0) # type: ignore

    if expect_list:
        # Convert the str to a list
        column: list[str] = literal_eval(str(info.loc[column_name].iloc[0])) # type: ignore

        print(f"Current value of {column_name}: {', '.join(column)}")
        new_value: str = input(f'Please enter the new value for {column_name}: ')

        # Update the DataFrame with the new value
        info.loc[column_name, info.columns[0]] = [item.strip() for item in new_value.split(',')]

    else:
        # Ask the user to input a new value
        print(f'Current value of {column_name}: {str(info.loc[column_name].iloc[0])}') # type: ignore

        # Update the DataFrame with the new value
        info.loc[column_name, info.columns[0]] = input(f'Please enter the new value for {column_name}: ')

    # Updates the CSV file
    info.to_csv(f"reports/{report_name}/info.csv") # type: ignore

    print(f'{column_name} updated successfully.')


def choose_stakeholder_to_modify(dm: DataManager) -> str:
    '''
    CLI tool to choose a stakeholder to modify.
    The function lists all the stakeholders stored in the report and prompts the user to select one by entering its corresponding number.
    The function returns the name of the selected stakeholder.

    Args:
        dm (DataManager): A valid DataManager object for the report.
    Returns:
        str: The name of the selected stakeholder.
    '''
    # Read the saved stakeholders
    saved_stakeholders = dm.load_data("stakeholders")
    all_saved_stakeholders = saved_stakeholders['direct_stakeholders'] + saved_stakeholders['indirect_stakeholders']

    # Asks the user to input a stakeholder and only continues when the user inputs a valid option
    stakeholder_dict = {no : stakeholder for no, stakeholder in enumerate(all_saved_stakeholders, start=1)}

    print('\nDirect stakeholders:')
    for no, stakeholder in enumerate(saved_stakeholders['direct_stakeholders'], start=1):
        if stakeholder in saved_stakeholders['direct_stakeholders']:
            print(f'{no}. {stakeholder}')

    print('\nIndirect stakeholders:')
    for no, stakeholder in enumerate(saved_stakeholders['indirect_stakeholders'], start=1 + len(saved_stakeholders['direct_stakeholders'])):
        if stakeholder in saved_stakeholders['indirect_stakeholders']:
            print(f'{no}. {stakeholder}')

    stakeholder_choice = 0
    while stakeholder_choice not in stakeholder_dict.keys():
        stakeholder_choice = input('\nPlease specify for which stakeholder you want to modify: ')

        if stakeholder_choice.isdigit():
            try:
                stakeholder_choice = int(stakeholder_choice)
            except ValueError:
                print('Input is not a number')

            if stakeholder_choice not in stakeholder_dict.keys():
                print('The provided number is not a valid option')
        else:
            print('Please input a number')

    chosen_stakeholder = str(stakeholder_dict[int(stakeholder_choice)])
    print(f'Chosen stakeholder: {chosen_stakeholder}')
    
    return chosen_stakeholder