import sys
import json



args = sys.argv[1:]
if len(args) < 2:
    print("Please specify the App mode.")
    exit(1)

app_mode = args[0]
if app_mode == "execute":
    from MacroExecute.MacroExecuter import MacroExecuter
    try:
        macro_name = args[1]
        with open(f"macro/{macro_name}/commands.json", "r", encoding='utf-8') as f:
            f_contents = json.load(f)
            macro_js = f_contents["commands"]
        executer = MacroExecuter()
        memory = {
            "macro_name": macro_name
        }
        executer.set_macro(macro_js)
        executer.set_memory(memory)
        executer.execute()
    except Exception as e:
        print(e)
        raise e
else:
    print(f"Invalid App mode.({args[0]})")
    exit(1)