
def closed(message: str) -> bool:
    """
    Asks the user for a yes or no confirmation.
    The function will keep asking until a valid answer is provided.
    It returns True for 'y' and False for 'n'.

    Args:
        message (str): The message to display to the user.
    Returns:
        bool: True if the user answered 'y', False if the user answered 'n'.
    """
    answer = ''
    while answer not in ('y', 'n'):
        answer = input(message + ' ').lower()
        if answer not in ('y', 'n'):
            print('Invalid choice. Please enter y or n.')
    
    if answer == 'y':
        return True
    else:
        return False