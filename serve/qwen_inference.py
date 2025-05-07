import json
from qwen_vl_utils import process_vision_info
from serve.system_prompts import open6dor_parsing_prompt, open6dor_reasoning_prompt


def open6dor_parsing(qwen_model, processor, image_path, instruction):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": open6dor_parsing_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": instruction},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = qwen_model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    output_text = output_text[0].replace('```json', '').replace('```', '')
    print(output_text)
    info = json.loads(output_text)
    directions = {}
    direction_attributes = []
    for direction in info['direction']:
        directions[direction['direction_attribute']] = direction['target_direction']
        direction_attributes.append(direction['direction_attribute'])
    info['target_orientation'] = directions
    info['direction_attributes'] = direction_attributes
    print(info)
    return info


def open6dor_spatial_reasoning(qwen_model, processor, image_path, instruction, picked_object_info, other_objects_info):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": open6dor_reasoning_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": f"Command: {instruction}\npicked_object_info: {picked_object_info}\nother_objects_info: {other_objects_info}"
                },
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = qwen_model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    output_text = output_text[0].replace('```json', '').replace('```', '')
    print(output_text)
    info = json.loads(output_text)
    print(info)
    return info
