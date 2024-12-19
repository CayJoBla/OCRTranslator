from PIL import Image, ImageDraw

def text_to_image(text, font):
    """Create an image from a line of text"""
    # Create a temporary image
    temp = Image.new("RGB", (1,1), "white")
    draw = ImageDraw.Draw(temp)

    # TODO: Determine the proper image size for the model
    # TODO: Depending on the above, maybe implement dynamic font size?

    # Get text size 
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Center the text
    x = -text_bbox[0]
    y = -text_bbox[1]

    # Add text to image
    image = Image.new("RGB", (text_width, text_height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((x, y), text, fill="black", font=font)

    return image

# def text_to_image(text, font, padding=0):
#     # Get text bounding box
#     text_bbox = font.getbbox(text)
#     text_width = text_bbox[2] - text_bbox[0]
#     text_height = text_bbox[3] - text_bbox[1]

#     # Add padding to the calculated text size
#     image_width = text_width + 2 * padding
#     image_height = text_height + 2 * padding

#     # Create the image
#     image = Image.new("RGB", (image_width, image_height), "white")
#     draw = ImageDraw.Draw(image)

#     # Draw text centered within the padded image
#     x = padding
#     y = padding
#     draw.text((x, y), text, fill="black", font=font)

#     return image

