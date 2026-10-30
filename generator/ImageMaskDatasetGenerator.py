# Copyright 2026 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ImageMaskDatasetGenerator.py


import numpy as np
import pandas as pd
import os
import cv2
import shutil
import traceback

class ImageMaskDataetGenerator:
  def __init__(self, augmentation=True):
    self.augmentation = augmentation
    self.hflip = True
    self.vflip = True
    #self.MASK_BGR_COLOR = (255, 255, 255)
    # Defined the dark red mask color. 
    self.MASK_BGR_COLOR = (20, 20, 210)


  def colorize_mask(self, mask,):
    h, w = mask.shape[:2]
    colorized = np.zeros((h, w, 3), dtype=np.uint8) 
    colorized[np.equal(mask, 1)] = self.MASK_BGR_COLOR 
    return colorized
  
  def normalize(self, image):
    img_min = image.min()
    img_max = image.max()
    
    if img_max - img_min != 0:
      normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
      normalized = np.zeros_like(image, dtype=np.uint8)
    return normalized
  
  def generate(self, index, filepath, images_dir, masks_dir):
    data = pd.read_pickle(filepath)
    print(data.shape)
    print(data.keys())
    #Index(['label1', 'mask', 'hu_array', 'hu_array_old'], dtype='object')

    self.generate_mask(index, data, masks_dir)
    self.generate_image(index, data, images_dir)
   

  def generate_mask(self, index, data, masks_dir):
    masks = data["mask"]
    print(masks.shape)
    num = masks.shape[0]
    print(num)
    
    for i in range(num):
      mask = masks[i]
      mask = self.colorize_mask(mask)

      index += 1
      print(mask.shape)
      filename = str(index ) + ".png"
      filepath = os.path.join(masks_dir, filename)
      cv2.imwrite(filepath, mask)
      print("Saved ", filepath)
  
      if self.augmentation:
        self.augment(mask, filename, masks_dir)

  def generate_image(self, index, data, images_dir):
    # Used images specified by the key "hu_array_old" not "hu_array".
    images = data["hu_array_old"]
    print(images.shape)
    num = images.shape[0]
    print(num)
    
    for i in range(num):
      image = images[i]
      image = self.normalize(image)
     
      index += 1
      print(image.shape)
      filename = str(index ) + ".png"
      filepath = os.path.join(images_dir, filename)
      cv2.imwrite(filepath, image )
      print("Saved ", filepath)
  
      if self.augmentation:
        self.augment(image, filename, images_dir)

  def augment(self, image, basename, output_dir):
    if self.hflip:
      flipped = self.horizontal_flip(image)
      output_filepath = os.path.join(output_dir, "hflipped_" + basename)
      cv2.imwrite(output_filepath, flipped)
      print("--- Saved {}".format(output_filepath))

    if self.vflip:
      flipped = self.vertical_flip(image)
      output_filepath = os.path.join(output_dir, "vflipped_" + basename)
      cv2.imwrite(output_filepath, flipped)
      print("--- Saved {}".format(output_filepath))


  def horizontal_flip(self, image): 
    print("shape image {}".format(image.shape))
    if len(image.shape)==3:
      return  image[:, ::-1, :]
    else:
      return  image[:, ::-1, ]

  def vertical_flip(self, image):
    if len(image.shape) == 3:
      return image[::-1, :, :]
    else:
      return image[::-1, :, ]


if __name__ == "__main__":
  try:
    output_dir = "./Augmented-Lung-Cancer-CT"
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    images_dir = "./Augmented-Lung-Cancer-CT/images"
    masks_dir  = "./Augmented-Lung-Cancer-CT/masks"
    os.makedirs(images_dir)
    os.makedirs(masks_dir)
    
    generator = ImageMaskDataetGenerator()

    filepath = "./lung_cancer_train.pkl"
    index = 1000
    generator.generate(index, filepath, images_dir, masks_dir)

    filepath = "./lung_cancer_test.pkl"
    index = 2000
    generator.generate(index, filepath, images_dir, masks_dir)

  except:
    traceback.print_exc()

