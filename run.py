import sys
from create_new_report import new_report
from modify_report import choose_report_to_modify, modify
from LLM import full_AFRIA, init_llm, get_available_models
import ask_confirmation


print('Welcome to the Automated Impact on Rights Assessment (AIRA) tool')
print('Do you want to create a new report or modify an existing one?')
print('1. Create a new report')
print('2. Modify an existing report')
print('3. Resume an interrupted report generation')
print('4. Exit\n')

choice = '0'
while choice not in range(1, 5):
    choice = input('Enter your choice: ')

    if choice.isdigit():
        choice = int(choice)
        if choice not in range(1, 5):
            print('Invalid choice. Please enter a valid option.')
    else:
        print('Please input a number')

if choice == 1:
    print('\nCreating a new report...')
    # Call the function to create a new report
    report_name = new_report()

    # Asks the user whether to run the full AFRIA process
    if ask_confirmation.closed('Do you want to run the full AFRIA process on this report (y/n)?'):
        # Asks the user whether it wants to use a custom LLM
        if ask_confirmation.closed('Do you want to use a custom LLM (default: Mistral Small) (y/n)?'):
            llm = init_llm(get_available_models())
            print('Successfully loaded the custom LLM.', end='\n\n')
        else:
            print('Using the default LLM.', end='\n\n')
            llm = init_llm()

        full_AFRIA(report_name, llm, resume=False)
    else:
        print('Exiting the program.')
        sys.exit(0)

elif choice == 2:
    print('\nModifying an existing report...')

    # Asks the user to choose a report to modify
    report_name = choose_report_to_modify()

    # Calls the function to modify the report
    modify(report_name)

elif choice == 3:
    print('\nResuming an interrupted report generation...')

    # Asks the user to choose a report to resume
    report_name = choose_report_to_modify()

    # Asks the user whether it wants to use a custom LLM
    if ask_confirmation.closed('Do you want to use a custom LLM (default: Mistral Small) (y/n)?'):
        llm = init_llm(get_available_models())
        print('Successfully loaded the custom LLM.', end='\n\n')
    else:
        print('Using the default LLM.', end='\n\n')
        llm = init_llm()

    # Calls the function to modify the report
    full_AFRIA(report_name, llm, resume=True)

elif choice == 4:
    print('\nExiting the program.')
    sys.exit(0)
