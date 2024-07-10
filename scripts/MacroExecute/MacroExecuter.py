from abc import ABC, abstractmethod
from .tools import *
import cv2
import numpy as np
from PIL import ImageGrab
import pyautogui
import time
import re
from .OriginalError import *
import copy
import osascript


################################ 実行に関わるクラス ################################
class CommandBlock:
    def __init__(self, command):
        self.commands:list[Command] = self.build_command_block(command)

    
    def build_command_block(self, command_block_js:list[dict[str, any]]):
        commands = []
        for command_dict in command_block_js:
            command = create_command(command_dict)
            commands.append(command)
        return commands
    

class BlockMemory:
    def __init__(self):
        self.main:dict[str, any] = {}
        self.env:dict[str, any] = {}

    def set_main(self, key:str, value:any):
        self.main[key] = value
    
    def get_main(self, key:str):
        return self.main[key]
    
    def has_main(self, key:str):
        return key in self.main
    
    def set_env(self, key:str, value:any):
        self.env[key] = value

    def get_env(self, key:str):
        return self.env[key]
    
    def has_env(self, key:str):
        return key in self.env
    
    def all_dict(self):
        return {**self.env, **self.main}
    
    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        new_copy = BlockMemory()
        memo[id(self)] = new_copy
        new_copy.main = copy.deepcopy(self.main, memo)
        new_copy.env = copy.deepcopy(self.env, memo)
        return new_copy


class MacroExecuter:
    def __init__(self, options:dict[str, any]={}):
        # self.hoge = options["hoge"]
        pass


    def set_macro(self, macro:CommandBlock):
        self.block = macro


    def set_memory(self, memory:BlockMemory):
        self.memory = memory


    def execute(self):
        for command in self.block.commands:
            status = command.execute(self.memory)

################################ コマンド抽象クラス ################################
class Command(ABC):
    def __init__(self, command, args, description):
        self.command:str = command
        self.args:list[dict[str, any]] = args
        self.description:str = description
    
    @abstractmethod
    def execute(self, memory:BlockMemory):
        pass
    
################################ コマンド実装 ################################

class Error(Command):
    def execute(self, memory:BlockMemory):
        print(self.args[0]["message"])
        exit(1)


class VisionClick(Command):
    def execute(self, memory:BlockMemory):
        # エラーチェック
        if len(self.args) != 1:
            raise SyntaxError("VisionClick command must have 1 arguments.")
        # 画像ファイルの読み込み
        template_path = f"macro/{memory.get_env('macro_name')}/PatternPictures/{self.args[0]['search_picture']}"
        template_img = cv2.imread(template_path, cv2.IMREAD_COLOR)
        target_img = cv2.cvtColor(np.array(ImageGrab.grab()), cv2.COLOR_RGB2BGR)

        # 画面キャプチャの取得
        best_matches = find_best_matches(template_img, target_img, threshold=float(self.args[0]["confidence"]))
        if best_matches == []:
            print("No matches found.")
            minimum_confidence = 0.5
            minimum_best_matches = find_best_matches(template_img, target_img, threshold=minimum_confidence)
            if minimum_best_matches == []:
                raise Exception(f"No matches found. by using minimum confidence({minimum_confidence}).")
            else:
                show_detected_img(minimum_best_matches, target_img)
                raise Exception(f"No matches found. by confidence({self.args[0]['confidence']}). However, by loosen confidence({minimum_confidence}), detected.")
        
        # クリック
        target_match = None
        if self.args[0]["select_axis"] == "":
            target_match = best_matches[0]
        elif self.args[0]["select_axis"] == "vertical":
            best_matches = sorted(best_matches, key=lambda x: x["pt"][1])
            target_match = best_matches[int(self.args[0]["select_index"])]
        elif self.args[0]["select_axis"] == "horizontal":
            best_matches = sorted(best_matches, key=lambda x: x["pt"][0])
            target_match = best_matches[self.args[0]["select_index"]]
        else:
            valid_axis = ["", "vertical", "horizontal"]
            raise ValueError(f"Unknown axis. Valid axis are {valid_axis}.")
        pt, size = target_match["pt"], target_match["size"]
        x, y = pt[0] + size[0] // 2, pt[1] + size[1] // 2
        time.sleep(0.01)
        abs_click(x, y)
        time.sleep(0.01)

        

class XYClick(Command):
    def execute(self, memory:BlockMemory):
        x, y = int(self.args[0]["x"]), int(self.args[0]["y"])
        abs_click(x, y)


class If(Command):
    def execute(self, memory:BlockMemory):
        # 式の変数を実際の値に置換
        args = self.args
        for i in range(len(args)):
            args[i]["replaced_conditions"] = replace_val(args[i]["condition"], memory.all_dict())
        # 各条件の評価
        for i in range(len(args)):
            args[i]["eval_result"] = eval(args[i]["replaced_conditions"])
        # 適切な分岐の選択
        for i in range(len(args)):
            if args[i]["eval_result"]:
                valid_block = args[i]["block"]
                break
        else:
            raise UnexpectedSystemError("All branches are invalid. Please check the conditions or set \"else block\".")
        # ブロックの実行
        branch_executer = MacroExecuter()
        branch_memory = copy.deepcopy(memory)
        branch_block = CommandBlock(valid_block)
        branch_executer.set_memory(branch_memory)
        branch_executer.set_macro(branch_block)
        branch_executer.execute()


class SetValue(Command):
    def execute(self, memory:BlockMemory):
        if len(self.args) != 1:
            raise SyntaxError("SetValue command must have exactly 1 argument.")
        memory.set_main(self.args[0]["variable_name"], self.args[0]["value"])


class Sleep(Command):
    def execute(self, memory:BlockMemory):
        if len(self.args) != 1:
            raise SyntaxError("Sleep command must have exactly 1 argument which is the time to sleep.")
        sleep_time = float(self.args[0]["time"])
        time.sleep(sleep_time)


################################ Chrome用コマンド ################################
class GetInfo(Command):
    def execute(self, memory: BlockMemory):
        # エラーチェック
        if len(self.args) != 1:
            raise SyntaxError("GetInfo command must have exactly one argument which decides what information is needed.")
        # 情報取得
        info_type = self.args[0]["information_type"]
        variable_name = self.args[0]["variable_name"]
        info = None
        target_img = cv2.cvtColor(np.array(ImageGrab.grab()), cv2.COLOR_RGB2BGR)
        separater_picture_path = memory.get_env('tab_separater_picture_path')
        current_tab_left_picture_path = memory.get_env('current_tab_left_picture_path')
        current_tab_right_picture_path = memory.get_env('current_tab_right_picture_path')
        first_tab_is_current_tab_picture_path= memory.get_env('first_tab_is_current_tab_picture_path')
        separater_picture = cv2.imread(separater_picture_path, cv2.IMREAD_COLOR)
        current_tab_left_picture = cv2.imread(current_tab_left_picture_path, cv2.IMREAD_COLOR)
        current_tab_right_picture = cv2.imread(current_tab_right_picture_path, cv2.IMREAD_COLOR)
        first_tab_is_current_tab_picture = cv2.imread(first_tab_is_current_tab_picture_path, cv2.IMREAD_COLOR)

        if info_type == "CurrentTabURL":
            info = get_current_tab_url()
        elif info_type == "CurrentTabIndex":
            info = get_current_tab_index(target_img, separater_picture, current_tab_left_picture, current_tab_right_picture, first_tab_is_current_tab_picture)
        else:
            valid_types = ["CurrentTabURL", "CurrentTabIndex"]
            raise ValueError(f"Unknown info label: {info_type}. Valid labels are {valid_types}.")
        # メモリに保存
        memory.set_main(variable_name, info)


class VisionWait(Command):
    def execute(self, memory:BlockMemory):
        # エラーチェック
        if len(self.args) != 1:
            raise SyntaxError("VisionWait command must have exactly two argument which decides the rest time of rejudge and the end time.")
        interval = float(self.args[0]["interval"])
        time_limit = float(self.args[0]["time_limit"])
        num_target = int(self.args[0]["num_target_object"])
        init_sleep_time = float(self.args[0]["init_sleep_time"])
        # 画像ファイルの読み込み
        template_path = f"macro/{memory.get_env('macro_name')}/PatternPictures/{self.args[0]['search_picture']}"
        template_img = cv2.imread(template_path, cv2.IMREAD_COLOR)
        time.sleep(init_sleep_time)
        for i in range(int(time_limit // interval)):
            target_img = cv2.cvtColor(np.array(ImageGrab.grab()), cv2.COLOR_RGB2BGR)
            best_matches = find_best_matches(template_img, target_img, threshold=float(self.args[0]["confidence"]))
            if num_target == len(best_matches):
                break
            time.sleep(interval)
        else:
            raise Exception("No matches found.")


class ChangeTab(Command):
    def execute(self, memory:BlockMemory):
        # エラーチェック
        if len(self.args) != 1:
            raise SyntaxError("ChangeTab command must have exactly 1 arguments which is the pair of tab information and the type of tab information.")
        info_type= self.args[0]["information_type"]
        tab_info = self.args[0]["tab_information"]
        # タブ移動
        valid_types = ["AbsoluteIndex", "RelativeIndex"]
        if info_type in valid_types:
            concreate_tab_info = int(replace_val(tab_info, memory.all_dict()))
            target_img = cv2.cvtColor(np.array(ImageGrab.grab()), cv2.COLOR_RGB2BGR)
            separater_picture_path = memory.get_env('tab_separater_picture_path')
            current_tab_left_picture_path = memory.get_env('current_tab_left_picture_path')
            current_tab_right_picture_path = memory.get_env('current_tab_right_picture_path')
            first_tab_is_current_tab_picture_path= memory.get_env('first_tab_is_current_tab_picture_path')
            separater_picture = cv2.imread(separater_picture_path, cv2.IMREAD_COLOR)
            current_tab_left_picture = cv2.imread(current_tab_left_picture_path, cv2.IMREAD_COLOR)
            current_tab_right_picture = cv2.imread(current_tab_right_picture_path, cv2.IMREAD_COLOR)
            first_tab_is_current_tab_picture = cv2.imread(first_tab_is_current_tab_picture_path, cv2.IMREAD_COLOR)
            change_tab(concreate_tab_info, info_type, target_img, separater_picture, current_tab_left_picture, current_tab_right_picture, first_tab_is_current_tab_picture)
        else:
            raise ValueError(f"Unknown tab decision: {info_type}. Valid decisions are {valid_types}.")




# コマンドの登録
command_list = {
    "Error" : Error,
    "VisionClick" : VisionClick,
    "XYClick" : XYClick,
    "If" : If,
    "SetValue" : SetValue,
    "GetInfo" : GetInfo,
    "VisionWait" : VisionWait,
    "ChangeTab" : ChangeTab,
    "Sleep" : Sleep
}


def create_command(command_dict:dict[str, any]):
    command = command_dict["command"]
    args = command_dict["args"]
    description = command_dict["description"]
    return command_list[command](command, args, description)
    