class UnexpectedSystemError(Exception):
    """論理上発生しないはずのエラー。"""
    pass


class OsaScriptError(Exception):
    """osascriptコマンドのエラー。"""
    pass