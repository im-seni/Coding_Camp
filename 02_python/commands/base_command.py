# commands/base_command.py
import os
from typing import List

"""
TODO 3-1: The BaseCommand class has a show_usage method implemented, but the execute method is not 
implemented and is passed on to the child class. Think about why this difference is made.

Answer (You may write your answer in either Korean or English): show_usage의 경우 자식 class 별로 변수만 다르게 적용되지만, 
execute는 자식 class별로 구현되는 기능 자체가 달라진다. 다양한 자식 클래스에서 서로 다른 기능을 구현할 수 있도록 보장하기 위해 
최소한의 공통 인터페이스만 부모 class에 구축했다고 생각한다. 

TODO 3-2: The update_current_path method of the BaseCommand class is slightly different from other methods. 
It has a @classmethod decorator and takes a cls argument instead of self. In Python, this is called a 
class method, and think about why it was implemented as a class method instead of a normal method.

Answer (You may write your answer in either Korean or English): @classmethod decorator를 가지고 있는 update_current_path는
현재 경로를 새로운 경로로 변경하는 기능을 수행하고 있다. 사용자가 cd 명령어를 통해 디렉토리를 이동하고자 할 경우, 현재 경로는 새로운 경로로 변경되어야 하며, 
이는 BaseCommand를 상속받고 있는 전체 클래스에 전달해 적용되어야 한다. 변경 사항은 특정 인스턴스가 아닌 클래스 전체에 공유 되어야 하므로, @classmethod를 사용했다고 볼 수 있다. 

"""
class BaseCommand:
    """
    Base class for all commands. Each command should inherit from this class and 
    override the execute() method.
    
    For example, the MoveCommand class overrides the execute() method to implement 
    the mv command.

    Attributes:
        current_path (str): The current path. Usefull for commands like ls, cd, etc.
    """

    current_path = os.getcwd()

    @classmethod
    def update_current_path(cls, new_path: str):
        """
        Update the current path.
        You need to understand how class methods work.

        Args:
            new_path (str): The new path. (Must be an relative path)
        """
        BaseCommand.current_path = os.path.join(BaseCommand.current_path, new_path)

    def __init__(self, options: List[str], args: List[str]) -> None:
        """
        Initialize a new instance of BaseCommand.

        Args:
            options (List[str]): The command options (e.g. -v, -i, etc.)
            args (List[str]): The command arguments (e.g. file names, directory names, etc.)
        """
        self.options = options
        self.args = args
        self.description = 'Helpful description of the command'
        self.usage = 'Usage: command [options] [arguments]'

    def show_usage(self) -> None:
        """
        Show the command usage.
        """
        print(self.description)
        print(self.usage)

    def execute(self) -> None:
        """
        Execute the command. This method should be overridden by each subclass.
        """
        raise NotImplementedError