import os
from PIL import Image
import PIL

PATH = r"D:\Python\gan\emoji_clean\image\\"
NEW_PATH = r"D:\Python\gan\emoji_clean_4\image"

for image in os.listdir(PATH):
  # im = Image.open(PATH + image)
  # # If is png image
  # if im.format == 'PNG':
  #   # and is not RGBA
  #   if im.mode != 'RGB':
  #     im.convert("RGB").save(NEW_PATH + image)
  
  
  im = Image.open(PATH + image)
  if list(im.getbands()) != ['R','G','B']:
      temp_background = PIL.Image.new('RGB', im.size, (255, 255, 255))
      temp_background.paste(im)
      im = temp_background
  im.save(os.path.join(NEW_PATH, image))
    
