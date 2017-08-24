from libs.utils import jar_wrapper
from libs import RESOURCES_DIR

def run_senti_strength(input_file, output_file=None):
    """
    Will run the parser for the SentiStrength to the input file
    Args:
        output_file: the output file of the converted file. If none, will
        input_file: name of the file that needs to

    Returns:

    """
    # read the input file
    with open(input_file, 'w') as f:
        # call the converter
        args = [RESOURCES_DIR + 'SentiStrength' + 'SentiStrength.jar', 'sentidata', './data', 'input', input_file]
        jar_wrapper(args)
