from abc import ABC, abstractmethod

@ABC
class Command:
    def __init__(self, command, target, args, description, code_blocks=[]):
        self.command:str = command
        self.target:str = target
        self.args:list[str] = args
        self.description:str = description
        self.code_blocks:list[str] = code_blocks
    
    @abstractmethod
    def execute(self, memory):
        pass
