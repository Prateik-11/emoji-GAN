import os
from PIL import Image
import PIL

# get absolute path of the file and pull images from there
PATH = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "image"), "Apple")
NEW_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clean_emojis")

for i, img_path in enumerate(os.listdir(PATH)):
  if img_path[-4:] == ".png":
    temp_image = Image.open(os.path.join(PATH, img_path))
    background = PIL.Image.new('RGBA', temp_image.size, (255, 255, 255))
    temp_image = temp_image.convert('RGBA')
    temp_image = PIL.Image.alpha_composite(background, temp_image).convert('RGB')
    temp_image.save(os.path.join(NEW_PATH, img_path))
    print(f"saved image: {img_path}", end='\r')
      
