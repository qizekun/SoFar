import os
import json
import base64
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def process_info(info):
    id = info["id"]

    base64_image = encode_image(f"datasets/6dof_spatialbench/images/{id}.png")
    question = info["question"]
    options = info["options"]
    answer = info["answer"]
    task_type = info["task_type"]
    question_type = info["question_type"]

    prompt = question + "\n" + "A. " + options[0] + "\n" + "B. " + options[1] + "\n" + "C. " + options[
        2] + "\n" + "D. " + options[3]
    messages = [
        {
            "role": "system",
            "content": """
                You are a spatial intelligence assistant specialized in understanding 3D visual scenes and answering spatial reasoning questions. 
                
                The user will provide:
                Image: An image of the scene.
                Question: User question about the spatial relationships between objects in the scene.
                
                Avoid providing answers such as "cannot determine." Instead, provide the most likely answer based on the information available.
                You will receive an image from the user, a question, and four options. 
                You only need to respond with A, B, C, or D without providing any additional information.
            """
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ],
        },
    ]

    for times in range(10):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=4096
            )
            text = completion.choices[0].message.content
            break
        except:
            print("Error")
            if times == 9:
                return False, task_type, question_type

    print(text)
    if ("A" == text[0] and answer == 0) or ("B" == text[0] and answer == 1) or ("C" == text[0] and answer == 2) or (
            "D" == text[0] and answer == 3):
        return True, task_type, question_type
    else:
        return False, task_type, question_type


if __name__ == "__main__":
    info_list = json.load(open('datasets/6dof_spatialbench/spatial_data.json')) * 5
    total = len(info_list)
    print("total: ", total)
    result = {
        "position": {
            "absolute": [],
            "relative": []
        },
        "orientation": {
            "absolute": [],
            "relative": []
        },
        "total": []
    }

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_info, info): info for info in info_list}
        for future in tqdm(as_completed(futures), total=total):
            flag, task_type, question_type = future.result()
            result["total"].append(flag)
            if task_type == "position":
                if question_type == "absolute":
                    result["position"]["absolute"].append(flag)
                else:
                    result["position"]["relative"].append(flag)
            else:
                if question_type == "absolute":
                    result["orientation"]["absolute"].append(flag)
                else:
                    result["orientation"]["relative"].append(flag)

    print("Position relative accuracy: ", sum(result["position"]["relative"]) / len(result["position"]["relative"]))
    print("Position absolute accuracy: ", sum(result["position"]["absolute"]) / len(result["position"]["absolute"]))
    print("Orientation relative accuracy: ",
          sum(result["orientation"]["relative"]) / len(result["orientation"]["relative"]))
    print("Orientation absolute accuracy: ",
          sum(result["orientation"]["absolute"]) / len(result["orientation"]["absolute"]))
    print("Total accuracy: ", sum(result["total"]) / len(result["total"]))
