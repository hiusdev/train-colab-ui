

from PIL import Image
import requests
import copy
import torch
import os



Caption_Length = "Long" # @param ["Short","Medium","Long"]
Cap_prompt = {
    'Short':['<CAPTION>',10,30,'short'],
    'Medium':['<DETAILED_CAPTION>',10,100,'medium-length'],
    'Long':['<MORE_DETAILED_CAPTION>',10,150,'very long']
}



def run_example(model,processor,task_prompt,image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer


def run_captions(model, processor, dir_data, dir_image):

    task_prompt = Cap_prompt[Caption_Length][0]

    # image_name = os.path.splitext(os.path.basename(img))[0]
    file_path = os.path.join(dir_data, dir_image)

    img = Image.open(file_path)
    # Define the caption or text content
    text = run_example(model,processor,task_prompt,img)
    # print(text[task_prompt])

    # Create the txt file path with the same name as the image
    image_name = dir_image.split(".")[0]

    txt_file_path = os.path.join(dir_data, f"{image_name}.caption")
    content = text[task_prompt]
    content = content.replace("\n", " ")
    content = content.replace("  ", " ")
    content = content.replace('The image shows','')
    content = content.replace('The image an','an')
    content = content.replace('The image a','a')
    content = content.replace('This is an','an')
    content = content.replace('This is a','a')

    # print(content)

    with open(txt_file_path, 'a') as txt_file:
        txt_file.write(content)
    return content
