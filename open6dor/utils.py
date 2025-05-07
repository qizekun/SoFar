from PIL import Image, ImageEnhance


def preprocess_open6dor_image(image):
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.5)
    width, height = image.size
    edge_width = int(width * 0.1)
    edge_height = int(height * 0.1)
    white_image = Image.new("RGB", (width, height), "white")
    white_image.paste(image.crop((edge_width, edge_height, width - edge_width, height - edge_height * 2)),
                      (edge_width, edge_height))
    return white_image
