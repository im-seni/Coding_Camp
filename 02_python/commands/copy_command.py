from .base_command import BaseCommand
import os
import shutil
from typing import List

class CopyCommand(BaseCommand):
    def __init__(self, options: List[str], args: List[str]) -> None:
        """
        Initialize the CopyCommand object.

        Args:
            options (List[str]): List of command options.
            args (List[str]): List of command arguments.
        """
        super().__init__(options, args)

        # Override the attributes inherited from BaseCommand
        self.description = 'Copy a file or directory to another location'
        self.usage = 'Usage: cp [source] [destination]'

        # TODO 6-1: Initialize any additional attributes you may need.
        # Refer to list_command.py, grep_command.py to implement this.
        # ...
        self.name = 'cp'
        self.options = options
        self.source  = self.args[0] if self.args else ''
        self.destination = self.args[1] if self.args else ''



    def execute(self) -> None:
        """
        Execute the copy command.
        Supported options:
            -i: Prompt the user before overwriting an existing file.
            -v: Enable verbose mode (print detailed information)
        
        TODO 6-2: Implement the functionality to copy a file or directory to another location.
        You may need to handle exceptions and print relevant error messages.
        You may use the file_exists() method to check if the destination file already exists.
        """
        # Your code here
        overwrite = '-i' in self.options
        info = '-v' in self.options
        new_path = os.path.join(self.destination, self.source)

        if info : 
            print(f"cp: copying '{self.source}' to '{self.destination}'")
        
        if self.file_exists(self.destination, self.source):
            if overwrite:
                user = input(f"cp: overwrite '{new_path}'? (y/n) ").strip().lower()
                if user == 'y':
                    shutil.copy(self.source, self.destination)
                else:
                    pass
            else:
                shutil.copy(self.source, self.destination)
        else:
                shutil.copy(self.source, self.destination)
                
        

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
