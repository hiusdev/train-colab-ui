import os
import numpy as np
import csv
import os
from onnxruntime import InferenceSession
from PIL import Image
import cv2


MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"
defaults = {
    "model": "wd-eva02-large-tagger-v3",
    "threshold": 0.35,
    "character_threshold": 0.85,
    "replace_underscore": False,
    "trailing_comma": False,
    "exclude_tags": "",
    "ortProviders": ["CUDAExecutionProvider", "CPUExecutionProvider"]
}


def tag(image,csv_path, model_path, threshold=0.35, character_threshold=0.85, exclude_tags="", replace_underscore=True, trailing_comma=False, client_id=None, node=None):

      model = InferenceSession(model_path, providers=defaults["ortProviders"])

      input = model.get_inputs()[0]
      height = input.shape[1]

      # Reduce to max size and pad with white
      ratio = float(height)/max(image.size)
      new_size = tuple([int(x*ratio) for x in image.size])
      image = image.resize(new_size, Image.LANCZOS)
      square = Image.new("RGB", (height, height), (255, 255, 255))
      square.paste(image, ((height-new_size[0])//2, (height-new_size[1])//2))

      image = np.array(square).astype(np.float32)
      image = image[:, :, ::-1]  # RGB -> BGR
      image = np.expand_dims(image, 0)

      # Read all tags from csv and locate start of each category
      tags = []
      general_index = None
      character_index = None
      with open(csv_path) as f:
          reader = csv.reader(f)
          next(reader)
          for row in reader:
              if general_index is None and row[2] == "0":
                  general_index = reader.line_num - 2
              elif character_index is None and row[2] == "4":
                  character_index = reader.line_num - 2
              if replace_underscore:
                  tags.append(row[1].replace("_", " "))
              else:
                  tags.append(row[1])

      label_name = model.get_outputs()[0].name
      probs = model.run([label_name], {input.name: image})[0]
      result = list(zip(tags, probs[0]))

      # rating = max(result[:general_index], key=lambda x: x[1])
      general = [item for item in result[general_index:character_index] if item[1] > threshold]
      character = [item for item in result[character_index:] if item[1] > character_threshold]

      all = character + general
      remove = [s.strip() for s in exclude_tags.lower().split(",")]
      all = [tag for tag in all if tag[0] not in remove]

      res = ("" if trailing_comma else ", ").join((item[0].replace("(", "\\(").replace(")", "\\)") + (", " if trailing_comma else "") for item in all))

      # print(res)
      return res



def numpy2pil(np_array: np.ndarray) -> Image:
    """
    Convert an HxWx3 numpy array into an RGB Image
    """

    assert_msg = 'Input shall be a HxWx3 ndarray'
    assert isinstance(np_array, np.ndarray), assert_msg
    assert len(np_array.shape) == 3, assert_msg
    assert np_array.shape[2] == 3, assert_msg

    img = Image.fromarray(np_array, 'RGB')
    return img

def run_tags(dir_data, dir_image, csv_path, model_path, general_threshold =  0.45, character_threshold = 0.35):

    file_path = os.path.join(dir_data, dir_image)
    im = cv2.imread(file_path)
    im= numpy2pil(im)
    if im.mode != "RGB":
        im = im.convert("RGB")

    additional_feature = tag(im,csv_path, model_path,threshold=general_threshold,character_threshold=character_threshold)
    # path_img = dir_image.split(".")[0]

    # # print(path_img +": "+ additional_feature)
    # with open(os.path.join(dir_data, path_img)+".txt", 'a') as the_file:
    #     the_file.write(','+additional_feature)
    return additional_feature
