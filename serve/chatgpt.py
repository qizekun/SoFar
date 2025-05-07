import os
import io
import time
import json
import base64
from PIL import Image
from openai import OpenAI
from serve.system_prompts import *

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# gemini-2.0-flash-exp is comparable and even better than the gpt-4o


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def encode_pil_image(image):
    format = 'PNG' if image.mode == 'RGBA' else 'JPEG'
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def long_horizon_planning(text, img):
    base64_image = encode_pil_image(img)
    messages = [
        {
            "role": "system",
            "content": long_horizon_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    },
                },
                {"type": "text", "text": text},
            ],
        }
    ]

    while True:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1024,
                timeout=30,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "steps",
                        "schema": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },
                        "strict": True
                    }
                }
            )
            response = completion.choices[0].message.content
            steps = json.loads(response)
            break
        except Exception as e:
            print("Retrying call ChatGPT", e)
            time.sleep(1)
    return steps


def object_parsing(text, img):
    base64_image = encode_pil_image(img)
    messages = [
        {
            "role": "system",
            "content": object_parsing_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    },
                },
                {"type": "text", "text": text},
            ],
        }
    ]

    while True:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1024,
                timeout=30,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "object_list",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "object_list": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                            },
                            "required": ["object_list"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
            )
            response = completion.choices[0].message.content
            object_list = json.loads(response)['object_list']
            break
        except Exception as e:
            print("Retrying call ChatGPT", e)
            time.sleep(1)
    return object_list


def vqa_parsing(text, img):
    base64_image = encode_pil_image(img)
    messages = [
        {
            "role": "system",
            "content": vqa_parsing_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    },
                },
                {"type": "text", "text": text},
            ],
        }
    ]
    while True:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=4096,
                timeout=30,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "vqa_info",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "analysis": {"type": "string"},
                                "info": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "object_name": {
                                                "type": "string",
                                            },
                                            "direction_attributes": {
                                                "type": "array",
                                                "items": {
                                                    "type": "string",
                                                },
                                            }
                                        },
                                        "required": ["object_name", "direction_attributes"],
                                        "additionalProperties": False
                                    }
                                },
                            },
                            "required": ["analysis", "info"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
            )

            response = completion.choices[0].message.content
            response = json.loads(response)
            info = {}
            for obj in response['info']:
                info[obj['object_name']] = obj['direction_attributes']
            break
        except Exception as e:
            print("Retrying call ChatGPT", e)
            time.sleep(1)
    return info


def manip_parsing(text, img):
    base64_image = encode_pil_image(img)
    messages = [
        {
            "role": "system",
            "content": manip_parsing_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    },
                },
                {"type": "text", "text": text},
            ],
        }
    ]

    while True:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=4096,
                timeout=30,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "manipulation_info",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "analysis": {"type": "string"},
                                "info": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "object_name": {
                                                "type": "string",
                                            },
                                            "direction_attributes": {
                                                "type": "array",
                                                "items": {
                                                    "type": "string",
                                                },
                                            }
                                        },
                                        "required": ["object_name", "direction_attributes"],
                                        "additionalProperties": False
                                    }
                                },
                            },
                            "required": ["analysis", "info"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
            )

            response = completion.choices[0].message.content
            response = json.loads(response)
            info = {}
            for obj in response['info']:
                info[obj['object_name']] = obj['direction_attributes']
            break
        except Exception as e:
            print("Retrying call ChatGPT", e)
            time.sleep(1)
    return info


def open6dor_parsing(text, img):
    base64_image = encode_pil_image(img)
    messages = [
        {
            "role": "system",
            "content": open6dor_parsing_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    },
                },
                {"type": "text", "text": text},
            ],
        }
    ]

    while True:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=4096,
                timeout=30,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "pick_and_place_info",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "picked_object": {"type": "string"},
                                "direction_pointing": {"type": "string"},
                                "direction": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "direction_attribute": {
                                                "type": "string",
                                            },
                                            "target_direction": {
                                                "type": "array",
                                                "items": {
                                                    "type": "number",
                                                },
                                            }
                                        },
                                        "required": ["direction_attribute", "target_direction"],
                                        "additionalProperties": False
                                    }
                                },
                                "related_objects": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["picked_object", "direction_pointing", "direction", "related_objects"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
            )

            response = completion.choices[0].message.content
            info = json.loads(response)
            directions = {}
            direction_attributes = []
            for direction in info['direction']:
                directions[direction['direction_attribute']] = direction['target_direction']
                direction_attributes.append(direction['direction_attribute'])
            info['target_orientation'] = directions
            info['direction_attributes'] = direction_attributes
            del info['direction']
            break
        except Exception as e:
            print("Retrying call ChatGPT", e)
            time.sleep(1)
    return info


def open6dor_spatial_reasoning(img, instruction, picked_object_info, other_objects_info):
    picked_object_info = json.dumps(picked_object_info, indent=2)
    other_objects_info = json.dumps(other_objects_info, indent=2)

    base64_image = encode_pil_image(img)
    messages = [
        {
            "role": "system",
            "content": open6dor_reasoning_prompt,
        },
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": f"Command: {instruction}\npicked_object_info: {picked_object_info}\nother_objects_info: {other_objects_info}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    },
                },
            ],
        }
    ]

    while True:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=4096,
                timeout=30,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "place_info",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "calculation_process": {"type": "string"},
                                "target_position": {
                                    "type": "array",
                                    "items": {"type": "number"}
                                }
                            },
                            "required": ["calculation_process", "target_position"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
            )

            response = completion.choices[0].message.content
            response = json.loads(response)
            break
        except Exception as e:
            print("Retrying call ChatGPT", e)
            time.sleep(1)

    return response


def manip_spatial_reasoning(img, instruction, scene_graph):
    scene_graph = json.dumps(scene_graph, indent=2)
    base64_image = encode_pil_image(img)
    messages = [
        {
            "role": "system",
            "content": manip_reasoning_prompt,
        },
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": f"Command: {instruction}\nScene Graph: {scene_graph}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    },
                },
            ],
        }
    ]

    while True:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=4096,
                timeout=30,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "manipulation_info",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "interact_object_id": {
                                    "type": "number"
                                },
                                "calculation_process": {
                                    "type": "string",
                                },
                                "target_position": {
                                    "type": "array",
                                    "items": {"type": "number"}
                                },
                                "target_orientation": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "direction_attributes": {
                                                "type": "string",
                                            },
                                            "value": {
                                                "type": "array",
                                                "items": {
                                                    "type": "number",
                                                },
                                            }
                                        },
                                        "required": ["direction_attributes", "value"],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            "required": ["interact_object_id", "calculation_process", "target_position",
                                         "target_orientation"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
            )

            response = completion.choices[0].message.content
            response = json.loads(response)
            response["target_orientation"] = {item["direction_attributes"]: item["value"] for item in
                                              response["target_orientation"]}
            break
        except Exception as e:
            print("Retrying call ChatGPT", e)
            time.sleep(1)

    return response


def vqa_spatial_reasoning(img, instruction, scene_graph, eval=False):
    scene_graph = json.dumps(scene_graph, indent=2)
    base64_image = encode_pil_image(img)
    messages = [
        {
            "role": "system",
            "content": vqa_reasoning_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    },
                },
                {"type": "text",
                 "text": f"Question: {instruction}\nScene graph: {scene_graph}\nAnswer:"},
            ],
        }
    ]

    if eval:
        messages[0]["content"] += ("You will receive an image from the user, a question, and four options. You only "
                                   "need to respond with A, B, C, or D without providing any additional information.")
    else:
        messages[0]["content"] += "Provide a detail analysis of the question and the image to answer the question."

    while True:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=4096,
                timeout=30
            )

            response = completion.choices[0].message.content
            break
        except Exception as e:
            print("Retrying call ChatGPT", e)
            time.sleep(1)

    return response


if __name__ == "__main__":
    text = ("Place the mobile phone behind the organizer on the table. We also need to specify the rotation of the "
            "object after placement:  the rectangular phone is placed steadily on the table, with the screen facing "
            "up, and is oriented in a 'pointing forward' way so that from a viewer's perspective, its two long edges "
            "are in a left-right position while the two short edges are in a front-back / near-far position.")
    SOURCE_IMAGE_PATH = "assets/open6dor.png"
    image = Image.open(SOURCE_IMAGE_PATH)
    info = open6dor_parsing(text, image)
    print(info)

