from tools import *
from PIL import ImageGrab
target_points = [(1133,862), (1148, 978), (1847, 333), (261, 1822), (223, 1822)]
target_img = ImageGrab.grab()
target_img = np.array(target_img)
gray_target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
for point in target_points:
    show_point(point, gray_target_img)