import polyscope.imgui as imgui

danger          = (205/255, 50/255, 50/255, 1.0)
danger_hover    = (205/255, 85/255, 85/255, 1.0)
subtle          = (120/255, 120/255, 120/255, 1.0)
subtle_hover    = (150/255, 150/255, 150/255, 1.0)

def default_button_size():      
    return [imgui.GetWindowWidth() / 2.1, 0]

def half_button_size():      
    return [imgui.GetWindowWidth() / (2.1*2), 0]

def small_button_size():
    return [imgui.GetWindowWidth() / (2.1*7), 0]

def default_tab_flags():
    return (imgui.ImGuiTabBarFlags_Reorderable 
            | imgui.ImGuiTabBarFlags_FittingPolicyScroll 
            | imgui.ImGuiTabBarFlags_TabListPopupButton)

def delete_key():
    return "X"

def set_style_danger():
    imgui.PushStyleColor(imgui.ImGuiCol_Button, danger)
    imgui.PushStyleColor(imgui.ImGuiCol_Tab, danger)

    imgui.PushStyleColor(imgui.ImGuiCol_ButtonHovered, danger_hover)
    imgui.PushStyleColor(imgui.ImGuiCol_TabHovered, danger_hover)
    imgui.PushStyleColor(imgui.ImGuiCol_TabActive, danger_hover)


def set_style_subtle():
    imgui.PushStyleColor(imgui.ImGuiCol_Button, subtle)
    imgui.PushStyleColor(imgui.ImGuiCol_Tab, subtle)

    imgui.PushStyleColor(imgui.ImGuiCol_ButtonHovered, subtle_hover)
    imgui.PushStyleColor(imgui.ImGuiCol_TabHovered, subtle_hover)
    imgui.PushStyleColor(imgui.ImGuiCol_TabActive, subtle_hover)
        
        
def pop_most_recent_style():
    imgui.PopStyleColor(5)