import sys
import json



args = sys.argv[1:]
if len(args) < 2:
    print("Please specify the App mode.")
    exit(1)

app_mode = args[0]
if app_mode == "execute":
    from MacroExecute.MacroExecuter import MacroExecuter
    from MacroExecute.MacroExecuter import BlockMemory
    from MacroExecute.MacroExecuter import CommandBlock
    def _init_memory(memory:BlockMemory, macro_name:str, device:str="Mac", main_window:str="hoge"):
        memory.set_env("macro_name", macro_name)
        memory.set_env("device", device)
        memory.set_env("main_window", main_window)
        return memory
    
    def _init_memory_for_chrome(memory:BlockMemory):
        memory.set_env("tab_separater_picture_path", f"macro/{memory.get_env('macro_name')}/PatternPictures/ChromeTabSeparater.png")
        memory.set_env("current_tab_left_picture_path", f"macro/{memory.get_env('macro_name')}/PatternPictures/ChromeCurrentTabLeft.png")
        memory.set_env("current_tab_right_picture_path", f"macro/{memory.get_env('macro_name')}/PatternPictures/ChromeCurrentTabRight.png")
        memory.set_env("first_tab_is_current_tab_picture_path", f"macro/{memory.get_env('macro_name')}/PatternPictures/ChromeFirstTabIsCurrentTab.png")
        return memory
    
    macro_name = args[1]
    with open(f"macro/{macro_name}/commands.json", "r", encoding='utf-8') as f:
        f_contents = json.load(f)
        macro_js = f_contents["commands"]
    executer = MacroExecuter()
    macro = CommandBlock(macro_js)
    memory = BlockMemory()
    memory = _init_memory(memory, macro_name)
    memory = _init_memory_for_chrome(memory)

    
    executer.set_macro(macro)
    executer.set_memory(memory)
    executer.execute()

else:
    print(f"Invalid App mode.({args[0]})")
    exit(1)