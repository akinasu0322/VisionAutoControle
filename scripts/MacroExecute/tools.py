import cv2
import numpy as np
from PIL import ImageGrab
import pyautogui
import json
import osascript
from .OriginalError import *
import re

cnf = json.load(open('scripts/config.json'))
device = cnf['device']
template_path = '../macro/PaperAutoSavePictures/test3.png'

# テンプレートマッチングで最も一致する結果を返す関数
"""
input:
    gray_template_img: テンプレート画像のグレースケール画像
    gray_target_img: 検出対象画像のグレースケール画像
    threshold: 一致と判定する信頼度の閾値
    scales: テンプレート画像の拡大率のリスト
    proximity: 近接したマッチング結果をグループ化する距離
return:
    best_matches: [(pt, confidence, scale, w, h), ...]
    一致した部分の情報を格納したリスト（sorted by confidence）
"""
def find_best_matches(gray_template_img, gray_target_img, threshold=0.9, scales=[0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0], proximity=10):
    matches = []

    for scale in scales:
        # テンプレートのサイズを変更
        resized_template = cv2.resize(gray_template_img, (0, 0), fx=scale, fy=scale)
        w, h = resized_template.shape[::-1]

        # テンプレートマッチングの実行
        result = cv2.matchTemplate(gray_target_img, resized_template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)

        # 一致する場所と信頼度を収集
        for pt in zip(*locations[::-1]):
            confidence = result[pt[1], pt[0]]
            matches.append((pt, confidence, scale, w, h))

    # 近接したマッチング結果をグループ化し、最も高い信頼度の結果を選択
    best_matches = []
    while matches:
        best_match = max(matches, key=lambda x: x[1])
        matches = [m for m in matches if np.linalg.norm(np.array(best_match[0]) - np.array(m[0])) > proximity]
        best_matches.append(best_match)

    return best_matches


# 絶対座標でクリックする関数
def abs_click(x, y, device=device):
    if device == "Mac":
        x, y = x // 2, y // 2
    pyautogui.click(x, y)


# 検出結果を表示する関数
def show_detected_img(best_matches, target_img):
    for (pt, confidence, scale, w, h) in best_matches:
        print(f"Location: {pt}, Confidence: {confidence}, Scale: {scale}, w: {w}, h: {h}")
        cv2.rectangle(target_img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
    cv2.imshow('Detected', target_img)
    print("エンターキーを押すとプログラムが終了します...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 変数を置換する関数
def replace_val(target:str, val_dict:dict[str, str]):
    # 空白除去
    pattern = re.compile(r"\$\{\s*([^\s, ^}]*?)\s*\}")
    clean_target = pattern.sub(lambda m: f"${{{m.group(1).strip()}}}", target)
    # 変数置換
    pattern = re.compile(r"\$\{(.*?)\}")
    def replace_one_val(m):
        val_name = m.group(1)
        val = str(val_dict[val_name])
        return val
    replaced_condition = pattern.sub(replace_one_val, clean_target)
    return replaced_condition


############################### Chrome操作用のツール ########################################
# 現在のタブのタブIDを取得する関数
def get_current_tab_id():
    script = """
    tell application "Google Chrome"
        if (count of windows) > 0 then
            set currentTabID to id of active tab of front window
        else
            set currentTabID to "No active Chrome window found."
        end if
    end tell
    return currentTabID
    """
    status, result, error = osascript.run(script)
    if status != 0:
        raise OsaScriptError(f"Error in osascript: {error}")
    if result == "No active Chrome window found.":
        raise OsaScriptError("No active Chrome window found.")
    return result


# 現在のタブのURLを取得する関数
def get_current_tab_url():
    script = """
    tell application "Google Chrome"
        if (count of windows) > 0 then
            set currentURL to URL of active tab of front window
        else
            set currentURL to "No active Chrome window found."
        end if
    end tell
    return currentURL
    """
    status, result, error = osascript.run(script)
    if status != 0:
        raise OsaScriptError(f"Error in osascript: {error}")
    if result == "No active Chrome window found.":
        raise OsaScriptError("No active Chrome window found.")
    return result

# 現在のwindowのchromeブラウザのタブIDを取得する関数
def get_open_tabs():
    """
    tab_list: [{tab_id, tab_url, ...}, ...]
        tab_id: int
        tab_url: str
    """
    script = """
    tell application "Google Chrome"
        set tabList to {}
        repeat with currentWindow in every window
            repeat with currentTab in every tab of currentWindow
                set currentTabID to id of currentTab
                set currentTabURL to URL of currentTab
                set currentTabInfo to {currentTabID, currentTabURL}
                set tabList to tabList & {currentTabInfo}
            end repeat
        end repeat
    end tell
    return tabList
    """
    status, result, error = osascript.run(script)
    if status != 0:
        raise OsaScriptError(f"Error in osascript: {error}")
    
    def transform_tab_data(data):
        items = data.split(', ')
        tab_info_list = []
        for i in range(0, len(items), 2):
            tab_id = items[i]
            tab_url = items[i + 1]
            tab_info_list.append({"tab_id": tab_id, "tab_url": tab_url})
        return tab_info_list
    
    tab_list = transform_tab_data(result)
    return tab_list


# タブのIDからタブの絶対インデックスを取得する関数
def get_absolute_tab_index(tab_id, tab_list):
    for i, tab_info in enumerate(tab_list):
        if tab_info["tab_id"] == tab_id:
            return i
    raise ValueError(f"Tab with ID {tab_id} not found.")


# タブの切り替えを行う関数
def change_tab(tab_information:int, info_type: str):
    def change_tab_by_tab_id(tab_id):
        script = f"""
        -- 事前に取得したタブIDを設定
        set targetTabID to {tab_id}

        tell application "Google Chrome"
            -- 現在開いているすべてのウィンドウを取得
            set windowList to every window
            repeat with currentWindow in windowList
                -- 現在のウィンドウのすべてのタブを取得
                set tabList to every tab of currentWindow
                repeat with currentTab in tabList
                    -- 現在のタブのIDを取得
                    set currentTabID to id of currentTab
                    -- タブIDが一致するか確認
                    if currentTabID is targetTabID then
                        -- 一致するタブをアクティブに設定
                        set active tab index of currentWindow to (index of currentTab)
                        -- スクリプトを終了
                        return "Tab switched to the target tab with ID " & targetTabID
                    end if
                end repeat
            end repeat
            return "Target tab with ID " & targetTabID & " not found."
        end tell
        """
        status, result, error = osascript.run(script)
        if status != 0:
            raise OsaScriptError(f"Error in osascript: {error}")

    def change_tab_by_absolute_tab_index(tab_index):
        tab_list = get_open_tabs()
        if tab_index >= len(tab_list) or tab_index < 0:
            raise ValueError(f"Absolute tab index {tab_index} is out of range.")
        tab_id = tab_list[tab_index]["tab_id"]
        change_tab_by_tab_id(tab_id)
    
    def change_tab_by_relative_tab_index(relative_index):
        tab_list = get_open_tabs()
        if relative_index >= len(tab_list) or relative_index < 0:
            raise ValueError(f"Relative tab index {relative_index} is out of range.")
        current_tab_id = get_current_tab_id()
        current_tab_index = get_absolute_tab_index(current_tab_id, tab_list)
        target_tab_index = current_tab_index + relative_index
        if target_tab_index >= len(tab_list) or target_tab_index < 0:
            raise ValueError(f"Relative tab index {relative_index} is out of range.")
        target_tab_id = tab_list[target_tab_index]["tab_id"]
        change_tab_by_tab_id(target_tab_id)
    

    if info_type == "TabID":
        change_tab_by_tab_id(int(tab_information))
    elif info_type == "RelativeIndex":
        change_tab_by_relative_tab_index(int(tab_information))
    elif info_type == "AbsoluteIndex":
        change_tab_by_absolute_tab_index(int(tab_information))
    else:
        valid_info_types = ["TabID", "RelativeIndex", "AbsoluteIndex"]
        raise ValueError(f"Unknown tab information type: {info_type}. Valid decisions are {valid_info_types}.")

