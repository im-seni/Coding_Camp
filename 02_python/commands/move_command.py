from .base_command import BaseCommand
import os
import shutil
from typing import List

class MoveCommand(BaseCommand):
    def __init__(self, options: List[str], args: List[str]) -> None:
        """
        Initialize the MoveCommand object.

        Args:
            options (List[str]): List of command options.
            args (List[str]): List of command arguments.
        """
        super().__init__(options, args)

        # Override the attributes inherited from BaseCommand
        self.description = 'Move a file or directory to another location'
        self.usage = 'Usage: mv [source] [destination]'

        # TODO 5-1: Initialize any additional attributes you may need.
        # Refer to list_command.py, grep_command.py to implement this.
        # ...
        self.name = 'mv'
        self.options = options
        self.source  = self.args[0] if self.args else ''
        self.destination = self.args[1] if self.args else ''


    def execute(self) -> None:
        """
        Execute the move command.
        Supported options:
            -i: Prompt the user before overwriting an existing file.
            -v: Enable verbose mode (print detailed information)
        
        TODO 5-2: Implement the functionality to move a file or directory to another location.
        You may need to handle exceptions and print relevant error messages.
        """
        # Your code here
        overwrite = '-i' in self.options
        moving = '-v' in self.options

        head, tail = os.path.split(self.source)
        if head and tail:
            filename = tail
        else:
            filename = self.source

        new_path = os.path.join(self.destination, filename)

        if moving:
                print(f"mv: moving '{self.source}' to '{self.destination}'")
        
        if self.file_exists(self.destination, filename):
            if overwrite:
                user = input(f"mv: overwrite '{new_path}'? (y/n) ").strip().lower()
                if user == 'y':
                    shutil.move(self.source, self.destination)
                else:
                    pass
            else:
                print(f"mv: cannot move{self.source} to {self.destination} : Destination path '{new_path}' already exists")
                return 
              
        if not self.file_exists(self.current_path, filename):
            shutil.move(self.source, self.destination)

    
    def file_exists(self, directory: str, file_name: str) -> bool:
        """
        Check if a file exists in a directory.
        Feel free to use this method in your execute() method.

        Args:
            directory (str): The directory to check.
            file_name (str): The name of the file.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        file_path = os.path.join(directory, file_name)
        return os.path.exists(file_path)
