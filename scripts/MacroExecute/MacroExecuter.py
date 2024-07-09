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
    def __init__(self, command, target, args, description, code_blocks=[]):
        self.command:str = command
        self.target:str = target
        self.args:list[str] = args
        self.description:str = description
        self.code_blocks:list[str] = code_blocks
    
    @abstractmethod
    def execute(self, memory:BlockMemory):
        pass
    
################################ コマンド実装 ################################

class Error(Command):
    def execute(self, memory:BlockMemory):
        print(self.target)
        exit(1)


class VisionClick(Command):
    def execute(self, memory:BlockMemory):
        # エラーチェック
        if len(self.args) != 0 and len(self.args) != 2:
            raise SyntaxError("VisionClick command must have 0 or 2 arguments.")
        # 画像ファイルの読み込み
        template_path = f"macro/{memory.get_env['macro_name']}/PatternPictures/{self.target}"
        gray_template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        target_img = ImageGrab.grab()
        target_img = np.array(target_img)
        gray_target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

        # 画面キャプチャの取得
        best_matches = find_best_matches(gray_template_img, gray_target_img, threshold=0.7)
        if best_matches == []:
            print("No matches found.")
            raise Exception("No matches found.")
        
        # クリック
        target_match = None
        if len(self.args) == 0:
            target_match = best_matches[0]
        elif len(self.args) == 2:
            axis = self.args[0]
            index = int(self.args[1])
            if axis == "vertical":
                best_matches = sorted(best_matches, key=lambda x: x[0][1])
                target_match = best_matches[index]
            elif axis == "horizontal":
                best_matches = sorted(best_matches, key=lambda x: x[0][0])
                target_match = best_matches[index]
            else:
                valid_axis = ["vertical", "horizontal"]
                raise ValueError(f"Unknown axis. Valid axis are {valid_axis}.")
        pt, w, h = target_match["pt"], target_match["size"][0], target_match["size"][1]
        x, y = pt[0] + w // 2, pt[1] + h // 2
        print(f"Clicking at ({x}, {y})")
        time.sleep(0.01)
        abs_click(x, y)
        time.sleep(0.01)

        

class XYClick(Command):
    def execute(self, memory:BlockMemory):
        x, y = map(int, self.target.replace("\s", "").split(","))
        abs_click(x, y)


class If(Command):
    def execute(self, memory:BlockMemory):
        # 式の変数を実際の値に置換
        conditions = self.args
        replaced_conditions = []
        for condition in conditions:
            replaced_conditions.append(replace_val(condition, memory.all_dict()))
        # 各条件の評価
        eval_results = []
        for condition in replaced_conditions:
            eval_results.append(eval(condition))
        eval_results.append(True) # elseに対応
        # 適切な分岐の選択
        for i, result in enumerate(eval_results):
            if result:
                valid_block = self.code_blocks[i]
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
            raise SyntaxError("SetValue command must have exactly one argument.")
        memory.set_main(self.target, self.args[0])


class Sleep(Command):
    def execute(self, memory:BlockMemory):
        if len(self.args) != 1:
            raise SyntaxError("Sleep command must have exactly 1 argument which is the time to sleep.")
        sleep_time = float(self.args[0])
        time.sleep(sleep_time)


################################ Chrome用コマンド ################################
class GetInfo(Command):
    def execute(self, memory: BlockMemory):
        # エラーチェック
        if len(self.args) != 1:
            raise SyntaxError("GetInfo command must have exactly one argument which decides what information is needed.")
        # 情報取得
        info_label = self.args[0]
        save_name = self.target
        info = None
        if info_label == "CurrentTabURL":
            info = get_current_tab_url()
        elif info_label == "CurrentTabID":
            info = get_current_tab_id()
        else:
            valid_labels = ["CurrentTabURL", "CurrentTabID"]
            raise ValueError(f"Unknown info label: {info_label}. Valid labels are {valid_labels}.")
        # メモリに保存
        memory.set_main(save_name, info)


class VisionWait(Command):
    def execute(self, memory:BlockMemory):
        # エラーチェック
        if len(self.args) != 2:
            raise SyntaxError("VisionWait command must have exactly two argument which decides the rest time of rejudge and the end time.")
        rest_time = float(self.args[0])
        end_time = float(self.args[1])
        # 画像ファイルの読み込み
        template_path = f"macro/{memory.main['macro_name']}/PatternPictures/{self.target}"
        gray_template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        for _ in range(int(end_time // rest_time)):
            target_img = ImageGrab.grab()
            target_img = np.array(target_img)
            gray_target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
            best_matches = find_best_matches(gray_template_img, gray_target_img, threshold=0.7)
            if best_matches:
                break
            time.sleep(rest_time)
        else:
            raise Exception("No matches found.")


class ChangeTab(Command):
    def execute(self, memory:BlockMemory):
        # エラーチェック
        if len(self.args) != 1:
            raise SyntaxError("ChangeTab command must have exactly 1 arguments which is the pair of tab information and the type of tab information.")
        info_type= self.args[0]["mode"]
        tab_info = self.args[0]["value"]
        # タブ移動
        valid_modes = ["AbsoluteIndex", "RelativeIndex"]
        if info_type in valid_modes:
            concreate_tab_info = int(replace_val(tab_info, memory.all_dict()))
            target_img = ImageGrab.grab()
            target_img = np.array(target_img)
            gray_target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
            separater_picture_path = memory.get_env('tab_separater_picture_path')
            current_tab_left_picture_path = memory.get_env('current_tab_left_picture_path')
            current_tab_right_picture_path = memory.get_env('current_tab_right_picture_path')
            first_tab_is_current_tab_picture_path= memory.get_env('first_tab_is_current_tab_picture_path')
            separater_picture = cv2.imread(separater_picture_path, cv2.IMREAD_GRAYSCALE)
            current_tab_left_picture = cv2.imread(current_tab_left_picture_path, cv2.IMREAD_GRAYSCALE)
            current_tab_right_picture = cv2.imread(current_tab_right_picture_path, cv2.IMREAD_GRAYSCALE)
            first_tab_is_current_tab_picture = cv2.imread(first_tab_is_current_tab_picture_path, cv2.IMREAD_GRAYSCALE)
            change_tab(concreate_tab_info, info_type, gray_target_img, separater_picture, current_tab_left_picture, current_tab_right_picture, first_tab_is_current_tab_picture)
        else:
            raise ValueError(f"Unknown tab decision: {info_type}. Valid decisions are {valid_modes}.")




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
    target = command_dict["target"]
    args = command_dict["args"]
    description = command_dict["description"]
    blocks = command_dict["blocks"]
    return command_list[command](command, target, args, description, blocks)
    