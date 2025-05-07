import json
import torch

from sofar_llava.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from sofar_llava.llava.conversation import conv_templates, SeparatorStyle
from sofar_llava.llava.model.builder import load_pretrained_model
from sofar_llava.llava.utils import disable_torch_init
from sofar_llava.llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path


def get_model(model_path, model_base, load_8bit=False, load_4bit=False, device="cuda"):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len \
        = load_pretrained_model(model_path, model_base, model_name, load_8bit, load_4bit, device=device)

    return model, tokenizer, image_processor


def inference(model, tokenizer, image_processor, image, inp):

    image_size = image.size
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()

    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    else:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=True,
            temperature=0.5,
            max_new_tokens=1024,
            use_cache=True)

    outputs = tokenizer.decode(output_ids[0]).strip()[3:-4]
    outputs = json.loads(outputs)
    return outputs
