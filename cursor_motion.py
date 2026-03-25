import pyautogui

# move the cursor from the movement of the finger that is tracked from the mediapipe and opencv
def move_cursor(x, y):
    # get the screen size
    screen_width, screen_height = pyautogui.size()
    
    # move the cursor to the new position
    pyautogui.moveTo(x * screen_width, y * screen_height)
