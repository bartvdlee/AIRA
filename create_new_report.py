import os
import pandas as pd

def new_report() -> str:
    '''
    Creates a new report directory and initializes it with a CSV file containing the scenario and stakeholders.
    The report directory is created under the 'reports' directory with the name provided by the user.
    The CSV file is named 'info.csv' and contains the scenario and user-provided stakeholders.
    The function checks if the directory already exists and handles exceptions accordingly.
    The function returns the name of the newly created report.
    '''
    # Asks the user for a name for the new report
    name = input('Please provide a name for the new report: ')

    # Creates the new report directory if it does not already exist, otherwise raises an error
    try:
        os.makedirs(f"reports/{name}")
    except RuntimeError:
        print('Failed creating the new report. Please check whether a report with the same name already exists.')
    
    # Asks the user for a scenario and stakeholders
    scenario = input('Please provide a scenario for the new report: ')
    stakeholders = input('Please identify and provide possible stakeholders in this scenario (stakeholder1, stakeholder2, etc.): ')
    problematic_behaviours = input('Please identify and provide possible problematic behaviours in this scenario (behaviour1, behaviour2, etc.): ')
    # Harm dimensions are not implemented

    stakeholder_list: list[str] = [item.strip() for item in stakeholders.split(',')]
    behaviour_list: list[str] = [item.strip() for item in problematic_behaviours.split(',')]

    # Creates a pandas Series with the scenario and stakeholders
    report_info = pd.Series(index=['scenario', 'user_specified_stakeholders', 'problematic_behaviours'], data=[scenario, stakeholder_list, behaviour_list])

    # Checks if the Series is not empty and has the correct length
    assert(len(report_info.iloc[0]) > 0), "Scenario is empty"
    assert(len(report_info.iloc[1]) > 0), "User-specified stakeholders are empty"
    assert(len(report_info.iloc[2]) > 0), "Problematic behaviours are empty"
    assert(len(report_info) == 3), "Report information is not of the correct length"

    # Creates a CSV file with the report information
    try:
        report_info.to_csv(f"reports/{name}/info.csv")
        print('Successfully created the new report')
    except RuntimeError:
        print('Failed writing the report info')
    
    return name