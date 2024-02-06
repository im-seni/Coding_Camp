import argparse
import logging
from utils.command_handler import CommandHandler
from utils.command_parser import CommandParser

# TODO 1-1: Use argparse to parse the command line arguments (verbose and log_file).
# TODO 1-2: Set up logging and initialize the logger object.

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", type = bool, default =False)
parser.add_argument("--log_path", type = str, default = 'file_explorer.log')

args = parser.parse_args()

verbose = args.verbose
log_path = args.log_path

if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
        level='INFO', filename = log_path, filemode = 'a' )

logger = logging.getLogger(__name__)



command_parser = CommandParser(verbose)
handler = CommandHandler(command_parser)

while True:
    command = input(">> ")
    handler.execute(command)
    logger.info(f"Input command: {command}")