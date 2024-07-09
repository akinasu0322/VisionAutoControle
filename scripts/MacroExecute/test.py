import re

test = [(1, 2), (3, 4), (5, 6)]

test = list(zip(*test))[1]
print(test)