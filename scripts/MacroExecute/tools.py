import cv2
import numpy as np
from PIL import ImageGrab, Image
import pyautogui
import json
import osascript
from .OriginalError import *
import re
import time

cnf = json.load(open('scripts/config.json'))
device = cnf['device']
template_path = '../macro/PaperAutoSavePictures/test3.png'

# テンプレートマッチングで最も一致する結果を返す関数

def find_best_matches(template_img, target_img, threshold=0.9, scales=[1.0], proximity=10):
    """
    input:
        template_img: テンプレート画像のRGB画像
        target_img: 検出対象画像のRGB画像
        threshold: 一致と判定する信頼度の閾値
        scales: テンプレート画像の拡大率のリスト
        proximity: 近接したマッチング結果をグループ化する距離
    return:
        best_matches: [{"pt":pt, "confidence":confidence, "scale":scale, "size":(w, h)}, ...]
        一致した部分の情報を格納したリスト（sorted by confidence）
    """
    # PIL画像をNumPy配列に変換
    if isinstance(template_img, Image.Image):
        template_img = np.array(template_img)
    if isinstance(target_img, Image.Image):
        target_img = np.array(target_img)

    matches = []
    for scale in scales:
        # テンプレートのサイズを変更
        resized_template = cv2.resize(template_img, (0, 0), fx=scale, fy=scale)
        w, h = resized_template.shape[1], resized_template.shape[0]

        # テンプレートマッチングの実行
        results = []
        for channel in range(3):
            result_channel = cv2.matchTemplate(target_img[:,:,channel], resized_template[:,:,channel], cv2.TM_CCOEFF_NORMED)
            results.append(result_channel)
        combined_result = np.sum(results, axis=0) / 3.0
        locations = np.where(combined_result >= threshold)

        # 一致する場所と信頼度を収集
        for pt in zip(*locations[::-1]):
            confidence = combined_result[pt[1], pt[0]]
            matches.append({"pt": pt, "confidence": confidence, "scale": scale, "size": (w, h)})

    # 近接したマッチング結果をグループ化し、最も高い信頼度の結果を選択
    best_matches = []
    while matches:
        best_match = max(matches, key=lambda x: x["confidence"])
        matches = [m for m in matches if np.linalg.norm(np.array(best_match["pt"]) - np.array(m["pt"])) > proximity]
        best_matches.append(best_match)

    return best_matches


# 絶対座標でクリックする関数
def abs_click(x, y, device=device):
    if device == "Mac":
        x, y = x // 2, y // 2
    pyautogui.click(x, y)


# 検出結果を表示する関数
def show_detected_img(best_matches, target_img):
    if isinstance(target_img, Image.Image):
        target_img = np.array(target_img)
        
    for match in best_matches:
        pt = match["pt"]
        confidence = match["confidence"]
        scale = match["scale"]
        w, h = match["size"]
        print(f"Location: {pt}, Confidence: {confidence}, Scale: {scale}, w: {w}, h: {h}")
        cv2.rectangle(target_img, pt, (pt[0] + w, pt[1] + h), (154, 205, 50), 4)
    cv2.imshow('Detected', target_img)
    print("エンターキーを押すとプログラムが終了します...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 指定ポイントが対象画像のどの部分を指すかを表示する関数
def show_point(pt, target_img):
    cv2.circle(target_img, pt, 20, (154, 205, 50), 4)
    cv2.imshow('Detected', target_img)
    print("エンターキーを押すとプログラムが終了します...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 画像を表示する関数
def show_img(img):
    cv2.imshow('Detected', img)
    print("エンターキーを押すとプログラムが終了します...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 座標とサイズから中央座標を計算する関数
def get_good_point(pt, size, option):
    if option == "top_left":
        return pt
    elif option == "top_right":
        return (pt[0] + size[0], pt[1])
    elif option == "bottom_left":
        return (pt[0], pt[1] + size[1])
    elif option == "bottom_right":
        return (pt[0] + size[0], pt[1] + size[1])
    elif option == "center":
        return (pt[0] + size[0] // 2, pt[1] + size[1] // 2)
    elif option == "top_center":
        return (pt[0] + size[0] // 2, pt[1])
    elif option == "bottom_center":
        return (pt[0] + size[0] // 2, pt[1] + size[1])
    elif option == "left_center":
        return (pt[0], pt[1] + size[1] // 2)
    elif option == "right_center":
        return (pt[0] + size[0], pt[1] + size[1] // 2)
    else:
        valid_options = ["top_left", "top_right", "bottom_left", "bottom_right", "center", "top_center", "bottom_center", "left_center", "right_center"]
        raise ValueError(f"Unknown option: {option}. Valid options are {valid_options}.")


# 座標を並行移動させる関数
def get_parallel_point(base_pt, ralative_pt):
    return (base_pt[0] + ralative_pt[0], base_pt[1] + ralative_pt[1])


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


# 現在タブの絶対インデックスを取得する関数
def get_current_tab_index(target_img, separater_picture, current_tab_left_picture, current_tab_right_picture, first_tab_is_current_tab_picture):
    tab_list, current_tab_idx = get_open_tabs(target_img, separater_picture, current_tab_left_picture, current_tab_right_picture, first_tab_is_current_tab_picture)
    return current_tab_idx


# 現在のwindowのchromeブラウザのタブIDを取得する関数
def get_open_tabs(target_img, separater_picture, current_tab_left_picture, current_tab_right_picture, first_tab_is_current_tab_picture):
    """
    input:
        current_tab_picture: 現在のタブ固有の画像（これが含まれる部分を現在のタブとする）
        separater_pictures: タブの区切りを示す画像のリスト（これで区切られた区間がタブとして認識される）
        target_img: タブの位置情報の取得対象画像（スクリーンショット画像）
    return:
        tab_list: [(a1, b1), (a2, b2), ...] 全てのタブの位置情報のリスト
        current_tab_idx: idx リスト内における現在のタブのインデックス
    """
    # 現在タブの左右位置を取得
    best_matches = find_best_matches(current_tab_left_picture, target_img)
    current_tab_left_point = get_good_point(best_matches[0]["pt"], best_matches[0]["size"], "center")
    best_matches = find_best_matches(current_tab_right_picture, target_img)
    current_tab_right_point = get_good_point(best_matches[0]["pt"], best_matches[0]["size"], "center")
    # タブセパレータの位置情報を取得
    best_matches = find_best_matches(separater_picture, target_img)
    separater_points = [get_good_point(separater_point["pt"], separater_point["size"], "center") for separater_point in best_matches] + [current_tab_left_point, current_tab_right_point]
    error_margin = 10
    trimmed_separated_points = [separater_point for separater_point in separater_points if separater_point[1] < min(list(zip(*separater_points))[1])+error_margin] # 一番上の列にないものは除外
    separater_points = sorted(trimmed_separated_points, key=lambda x: x[0])
    # 一番左に現在タブがあるかどうかを判定
    best_matches = find_best_matches(first_tab_is_current_tab_picture, target_img, threshold=0.98)
    first_tab_is_current_tab_point = get_good_point(best_matches[0]["pt"], best_matches[0]["size"], "center") if best_matches else None

    # タブと紐づけるセパレータのリストを作成
    if first_tab_is_current_tab_point: # セパレータの左にあるものをタブとして認識したいので、一番左に現在タブがある場合current_tab_left_pointのセパレータは左にタブを持たないため無視したい
        assert separater_points[0][0] == current_tab_left_point[0]
        assert separater_points[1][0] == current_tab_right_point[0]
        tab_identify_separater_points = separater_points[1:]
    else: # 2番目以降に現在タブがある場合
        tab_identify_separater_points = separater_points
    # 現在タブのインデックスを取得
    current_tab_idx = None
    for i, point in enumerate(tab_identify_separater_points):
        if point == current_tab_right_point:
            current_tab_idx = i
            break
    else:
        raise UnexpectedSystemError("Current tab is not found in the tab list.")
    # タブのリストを作成
    adjusting_pt = (-10, 0)# タブセパレーターの何ピクセル左をタブとして認識するかを決める
    tab_list = [get_parallel_point(tab_identify_separater_point, adjusting_pt) for tab_identify_separater_point in tab_identify_separater_points]
    return tab_list, current_tab_idx

    
    


    # タブの総数が1つの場合(separated_pointsが2個)

    
    return tab_list


# タブの切り替えを行う関数
def change_tab(tab_information:int, info_type: str, target_img, separater_picture, current_tab_left_picture, current_tab_right_picture, first_tab_is_current_tab_picture):
    # タブの切り替えを行う
    def _change_tab_core(tab_list, absolute_index, current_tab_idx):
        if absolute_index >= len(tab_list) or absolute_index < -len(tab_list):
            raise ValueError(f"Absolute tab index {absolute_index} is out of range.")
        if absolute_index == current_tab_idx:
            return
        abs_click(tab_list[absolute_index][0], tab_list[absolute_index][1])
        time.sleep(0.1)

    def change_tab_by_absolute_tab_index(absolute_index, target_img, separater_picture, current_tab_left_picture, current_tab_right_picture, first_tab_is_current_tab_picture):
        tab_list, current_tab_idx = get_open_tabs(target_img, separater_picture, current_tab_left_picture, current_tab_right_picture, first_tab_is_current_tab_picture)
        _change_tab_core(tab_list, absolute_index, current_tab_idx)
    
    def change_tab_by_relative_tab_index(relative_index, target_img, separater_picture, current_tab_left_picture, current_tab_right_picture, first_tab_is_current_tab_picture):
        tab_list, current_tab_idx = get_open_tabs(target_img, separater_picture, current_tab_left_picture, current_tab_right_picture, first_tab_is_current_tab_picture)
        absolute_idx = current_tab_idx + relative_index
        if absolute_idx < 0:
            raise ValueError(f"Current tab index: {current_tab_idx}, Relative tab index: {relative_index}. Finally, absolute tab index is {absolute_idx}. It is out of range.")
        _change_tab_core(tab_list, absolute_idx, current_tab_idx)


    if info_type == "AbsoluteIndex":
        change_tab_by_absolute_tab_index(int(tab_information), target_img, separater_picture, current_tab_left_picture, current_tab_right_picture, first_tab_is_current_tab_picture)
    elif info_type == "RelativeIndex":
        change_tab_by_relative_tab_index(int(tab_information), target_img, separater_picture, current_tab_left_picture, current_tab_right_picture, first_tab_is_current_tab_picture)
    else:
        valid_info_types = ["RelativeIndex", "AbsoluteIndex"]
        raise ValueError(f"Unknown tab information type: {info_type}. Valid decisions are {valid_info_types}.")


def parse_range(range_str:str):
    if '-' in range_str:
        start, end = range_str.split('-')
        return (int(start), int(end))
    elif range_str.endswith('+'):
        start = range_str[:-1]
        return (int(start), None)
    elif range_str.isdigit():
        return (int(range_str), int(range_str))
    else:
        raise ValueError("Invalid range format")


def check_num_in_range(num_target_object:int, range_list:list[tuple[int, int]]):
    check_num_in_one_range = lambda num, start, end: (end is None and start <= num) or (start <= num <= end)
    if any([check_num_in_one_range(num_target_object, start, end) for start, end in range_list]):
        return True
    return False