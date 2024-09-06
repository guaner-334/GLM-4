from PIL import ImageGrab


def screenshot_full(uuid, chat_id):
    """
    截取全屏
    :param uuid:对话的uuid
    :param chat_id:对话id
    :return:返回截图的路径tools/file/pic/{chat_id}-{uuid}.png
    """
    # 截取全屏
    screenshot = ImageGrab.grab()
    screenshot.save(f'file/pic/{chat_id}-{uuid}.png')
    return f"tools/file/pic/{chat_id}-{uuid}.png"