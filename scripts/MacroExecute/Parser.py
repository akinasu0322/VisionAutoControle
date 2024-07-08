# class Parser:
#     def __init__(self, lexer):
#         self.lexer = lexer
#         self.current_token = self.lexer.get_next_token()

#     def error(self):
#         raise Exception('Invalid syntax')

#     def eat(self, token_type):
#         if self.current_token.type == token_type:
#             self.current_token = self.lexer.get_next_token()
#         else:
#             self.error()

#     def factor(self):
#         token = self.current_token
#         if token.type == INTEGER:
#             self.eat(INTEGER)
#             return token.value
#         elif token.type == LPAREN:
#             self.eat(LPAREN)
#             result = self.expr()
#             self.eat(RPAREN)
#             return result

#     def term(self):
#         result = self.factor()
#         while self.current_token.type in (MUL, DIV):
#             token = self.current_token
#             if token.type == MUL:
#                 self.eat(MUL)
#                 result = result * self.factor()
#             elif token.type == DIV:
#                 self.eat(DIV)
#                 result = result / self.factor()
#         return result

#     def expr(self):
#         result = self.term()
#         while self.current_token.type in (PLUS, MINUS):
#             token = self.current_token
#             if token.type == PLUS:
#                 self.eat(PLUS)
#                 result = result + self.term()
#             elif token.type == MINUS:
#                 self.eat(MINUS)
#                 result = result - self.term()
#         return result

#     def parse(self):
#         return self.expr()