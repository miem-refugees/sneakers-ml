from PIL import Image


def crop_image(image: Image.Image, crop_sides: int, crop_top_bot: int) -> Image.Image:
    width, height = image.size
    left = (width - crop_sides) // 2
    top = (height - crop_top_bot) // 2
    right = (width + crop_sides) // 2
    bottom = (height + crop_top_bot) // 2

    return image.crop((left, top, right, bottom))
