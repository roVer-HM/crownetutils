"""Utils for user interaction on the command line.

"""

import sys


def query_yes_no(question: str, default=None) -> bool:
    """Ask a yes/no question via raw_input() and return their answer.

    Source:
    `Trent Mick  <https://code.activestate.com/recipes/users/4173505/>`_ at
    `Activestate <https://code.activestate.com/recipes/577058/>`_

    Args:
        question (str): prompt text for user.
        default (str, optional): Default answer yes or no. If not set an answer is required. Defaults to None.

    Returns:
        bool: user input
    """
    valid = {"yes": True, "y": True, "ye": True, "j": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default:
        prompt = " [Y/n] "
    else:
        prompt = " [y/N] "

    while 1:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return default
        elif choice in valid.keys():
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")
