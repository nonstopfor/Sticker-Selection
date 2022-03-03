import imghdr
from io import BytesIO
from PIL import Image, ImageSequence
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# 解析图片二进制数据到PIL Image对象，如果是静态图片则直接读取，如果是gif则解析第一帧
def pick_single_frame(data: bytes) -> Image.Image:
    if judge_img_type(data) != 'gif':
        img = read_pil_image(data)
    else:
        img = get_gif_nframe(data, 0)  # 取gif的第0帧
    img = remove_transparency(img)
    img = img.convert('RGB')
    return img


# 从内存data读取PIL格式
def read_pil_image(data):
    img = Image.open(BytesIO(data))
    return img


# 从内存数据判断图像格式
# all types: https://docs.python.org/zh-cn/3/library/imghdr.html
def judge_img_type(data):
    img_type = imghdr.what(BytesIO(data))
    return img_type


# 从内存读取gif并存储每一帧, 返回一个Iterator
def get_gif_iter(data):
    if judge_img_type(data) != 'gif':
        return
    gif = Image.open(BytesIO(data))
    for frame in ImageSequence.Iterator(gif):
        img = frame.convert('RGBA')
        yield img


# 读取gif的第n帧, 从0开始计数
# n支持负数(表示倒数第|n|张)
def get_gif_nframe(data, n):
    if judge_img_type(data) != 'gif':
        return None
    gif = Image.open(BytesIO(data))
    if not -gif.n_frames <= n <= gif.n_frames-1:
        return None
    if n < 0:
        n = gif.n_frames + n
    gif.seek(n)
    img = gif.convert('RGBA')
    return img


# 删除alpha通道
def remove_transparency(img_pil, bg_colour=(255, 255, 255)):
    # Only process if image has transparency
    if img_pil.mode in ('RGBA', 'LA') or \
        (img_pil.mode == 'P' and 'transparency' in img_pil.info):
        # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
        alpha = img_pil.convert('RGBA').getchannel('A')

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        new_img = Image.new("RGBA", img_pil.size, bg_colour + (255,))
        new_img.paste(img_pil, mask=alpha)
        return new_img
    else:
        return img_pil