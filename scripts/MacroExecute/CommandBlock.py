from Command import Command


class CommandBlock:
    def __init__(self, command):
        self.commands:list[Command] = self.build_command_block(command)

    
    def build_command_block(self, command_block_js:list[dict[str, any]]):
        commands = []
        for command_dict in command_block_js:
            command = Command.new(command_dict["command"], command_dict["target"], command_dict["args"], command_dict["description"], command_dict["blocks"])
            commands.append(command)
        return commands