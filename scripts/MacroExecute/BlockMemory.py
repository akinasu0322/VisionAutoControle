


class BlockMemory:
    def __init__(self):
        self.blocks = []

    def add_block(self, block):
        self.blocks.append(block)

    def get_block(self, index):
        return self.blocks[index]

    def get_block_count(self):
        return len(self.blocks)