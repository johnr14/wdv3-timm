#!/usr/bin/env python3

from dataclasses import field, dataclass
from pathlib import Path
from typing import Optional, List, Dict

import json
import glob
import os

import numpy as np
import pandas as pd
import timm
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image
from simple_parsing import field, parse_known_args
from timm.data import create_transform, resolve_data_config
from torch import Tensor, nn
from torch.nn import functional as F

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_REPO_MAP = {
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
}

def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    # convert to RGB/RGBA if not already (deals with palette images etc.)
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    # convert RGBA to RGB with white background
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    # get the largest dimension so we can pad to a square
    px = max(image.size)
    # pad to square with white background
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas


@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]


def load_labels_hf(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> LabelData:
    try:
        csv_path = hf_hub_download(
            repo_id=repo_id, filename="selected_tags.csv", revision=revision, token=token
        )
        csv_path = Path(csv_path).resolve()
    except HfHubHTTPError as e:
        raise FileNotFoundError(f"selected_tags.csv failed to download from {repo_id}") from e

    df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
    tag_data = LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )

    return tag_data


def get_tags(
    probs: Tensor,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
):
    # Convert indices+probs to labels
    probs = list(zip(labels.names, probs.numpy()))

    # First 4 labels are actually ratings
    rating_labels = dict([probs[i] for i in labels.rating])

    # General labels, pick any where prediction confidence > threshold
    gen_labels = [probs[i] for i in labels.general]
    gen_labels = dict([x for x in gen_labels if x[1] > gen_threshold])
    gen_labels = dict(sorted(gen_labels.items(), key=lambda item: item[1], reverse=True))

    # Character labels, pick any where prediction confidence > threshold
    char_labels = [probs[i] for i in labels.character]
    char_labels = dict([x for x in char_labels if x[1] > char_threshold])
    char_labels = dict(sorted(char_labels.items(), key=lambda item: item[1], reverse=True))

    # Combine general and character labels, sort by confidence
    combined_names = [x for x in gen_labels]
    combined_names.extend([x for x in char_labels])

    # Convert to a string suitable for use as a training caption
    caption = ", ".join(combined_names)
    taglist = caption.replace("_", " ").replace("(", "\(").replace(")", "\)")

    return caption, taglist, rating_labels, char_labels, gen_labels

def extract_keys_for_summary(data, key_name: str):
    """Extracts 'General' keys and 'Repo_id' from JSON data."""
    extracted_keys = []
    for item in data:
        if key_name in item and 'Repo_id' in item:
            repo_id = item['Repo_id']
            for key, value in item[key_name].items():
                extracted_keys.append((repo_id, key, value))
    return extracted_keys


def merge_and_order_keys(general_keys):
    """Merges and orders 'General' keys while keeping duplicates."""
    # Convert to list of tuples, then sort
    sorted_keys = sorted(general_keys, key=lambda x: x[1])
    return sorted_keys


def remove_keys_with_pattern(general_keys, pattern):
    """Removes key-value pairs where key name contains pattern."""
    filtered_keys = [(repo_id, key, value) for repo_id, key, value in general_keys if pattern not in key]
    return filtered_keys


def get_highest_value_pairs(filtered_keys):
    """Returns key-value pairs for each unique key with its highest value and corresponding Repo_id."""
    highest_value_pairs = {}
    for repo_id, key, value in filtered_keys:
        try:
            # Try converting value to float for comparison
            float_value = float(value)
            if key not in highest_value_pairs or float_value > float(highest_value_pairs[key][2]):
                highest_value_pairs[key] = (repo_id, key, value)
        except ValueError:
            # Non-numeric value, keep it if it's the first occurrence
            if key not in highest_value_pairs:
                highest_value_pairs[key] = (repo_id, key, value)
    return highest_value_pairs.values()


def print_key_value_pairs(value_pairs, input_file):
    if save_to_json:
        output_file = f"{os.path.splitext(input_file)[0]}_tags.json"
        dict_value_pairs = {}
        for repo_id, key, value in value_pairs:
            if repo_id not in dict_value_pairs:
                dict_value_pairs[repo_id] = {}
            dict_value_pairs[repo_id][key] = value
        with open(output_file, 'w') as f:
            json.dump(dict_value_pairs, f, indent=4)
    else:
        """Prints each key-value pair on a separate line, including Repo_id."""
        print(f"{'Key':<40} {'Value':<20} {'Repo ID'}")
        print("-" * 100)

        for repo_id, key, value in value_pairs:
            print(f"{key:<40} {value:<20} {repo_id}")

def write_to_json(image_path: Path, caption: str, taglist: List[str], ratings: Dict[str, float], character: Dict[str, float], general: Dict[str, float], repo_id: str, summary: bool = False) -> None:

    # Create the full path for the json file
    json_file_path = get_json_filepath(image_path)
    print("JSON FILE: "+json_file_path)

    # Create the json data
    data = {
        "Caption": caption,
        "Tags": taglist,
        "Ratings": {k: str(v) for k, v in ratings.items()},
        "Character": {k: str(v) for k, v in character.items()},
        "General": {k: str(v) for k, v in general.items()},
        "Repo_id": repo_id
    }

    # Check if the json file already exists
    if os.path.exists(json_file_path):
        # Get the original file size
        original_size = os.path.getsize(json_file_path)

        # If it exists, load the existing data and append the new data
        with open(json_file_path, 'r') as f:
            existing_data = json.load(f)

        # Append the new data to the existing data
        existing_data.append(data)

        # summary will be True when it's time to write the last model's results to JSON
        if summary:
            pattern = ':>='  # pattern to remove

            general_keys = extract_keys_for_summary(existing_data, "General")
            ratings_keys = extract_keys_for_summary(existing_data, "Ratings")
            caracter_keys = extract_keys_for_summary(existing_data, "Character")

            #filtered_general_keys = remove_keys_with_pattern(general_keys, pattern)

            highest_general_keys = get_highest_value_pairs(remove_keys_with_pattern(general_keys, pattern))
            highest_ratings_keys = get_highest_value_pairs(ratings_keys)
            highest_caracter_keys = get_highest_value_pairs(caracter_keys)

#             dict_value_pairs = {}
#             dict_value_pairs["Summary"] = {}
#             for repo_id, key, value in highest_value_pairs:
#                 if repo_id not in dict_value_pairs["Summary"]:
#                     dict_value_pairs["Summary"][repo_id] = {}
#                 dict_value_pairs["Summary"][repo_id][key] = value
#
#             for repo_id, key, value in highest_ratings_keys:
#                 if "Ratings" not in dict_value_pairs["Summary"][repo_id]:
#                     dict_value_pairs["Summary"][repo_id]["Ratings"] = {}
#                 dict_value_pairs["Summary"][repo_id]["Ratings"][key] = value
#
#             for repo_id, key, value in highest_caracter_keys:
#                 if "Character" not in dict_value_pairs["Summary"][repo_id]:
#                     dict_value_pairs["Summary"][repo_id] = {"Character": set()}
#                 dict_value_pairs["Summary"][repo_id]["Character"][key] = value
#
#             for repo_id, key, value in highest_ratings_keys:
#                 if "Caption" not in dict_value_pairs["Summary"]["Caption"]:
#                     dict_value_pairs["Summary"]["Caption"] = {}
#                 dict_value_pairs["Summary"][repo_id]["Character"].append(key)
#
#             for repo_id, key, value in highest_ratings_keys:
#                 if "Tags" not in dict_value_pairs["Summary"]:
#                     dict_value_pairs["Summary"]["Tags"] = {}
#                 dict_value_pairs["Summary"][repo_id]["Character"][key] = value
#
            dict_value_pairs = {}
            dict_value_pairs["Summary"] = {}

            # Process highest value pairs
            for repo_id, key, value in highest_general_keys:
                if "General" not in dict_value_pairs["Summary"]:
                    dict_value_pairs["Summary"]["General"] = {}
                if "General_Models" not in dict_value_pairs["Summary"]:
                    dict_value_pairs["Summary"]["General_Models"] = {}
                if repo_id not in dict_value_pairs["Summary"]["General_Models"]:
                    dict_value_pairs["Summary"]["General_Models"][repo_id] = []
                # List of selected keys per repo_id
                dict_value_pairs["Summary"]["General_Models"][repo_id].append(key)
                # Best key and value selected
                dict_value_pairs["Summary"]["General"][key] = value

            # Process highest ratings keys
            ratings_set = set()
            for repo_id, key, value in highest_ratings_keys:
                # ratings_set.add(key)
                if "Ratings" not in dict_value_pairs["Summary"]:
                    dict_value_pairs["Summary"]["Ratings"] = {}
                dict_value_pairs["Summary"]["Ratings"][key] = value
                if "Ratings_Models" not in dict_value_pairs["Summary"]:
                    dict_value_pairs["Summary"]["Ratings_Models"] = {}
                if repo_id not in dict_value_pairs["Summary"]["Ratings_Models"]:
                    dict_value_pairs["Summary"]["Ratings_Models"][repo_id] = []
                # List of selected keys per repo_id
                dict_value_pairs["Summary"]["Ratings_Models"][repo_id].append(key)
                # Best key and value selected
                dict_value_pairs["Summary"]["Ratings"][key] = value

            # Process highest character keys
            for repo_id, key, value in highest_caracter_keys:
                if "Character" not in dict_value_pairs["Summary"]:
                    dict_value_pairs["Summary"]["Character"] = {}
                if "Character_Models" not in dict_value_pairs["Summary"]:
                    dict_value_pairs["Summary"]["Character_Models"] = {}
                if repo_id not in dict_value_pairs["Summary"]["Character_Models"]:
                    dict_value_pairs["Summary"]["Character_Models"][repo_id] = []
                # List of selected keys per repo_id
                dict_value_pairs["Summary"]["Character_Models"][repo_id].append(key)
                # Best key and value selected
                dict_value_pairs["Summary"]["Character"][key] = value

            # Process caption
            caption_set = set()
            for repo_id, key, value in highest_general_keys:
                caption_set.add(key.replace(" ", "_"))
                if "Caption" not in dict_value_pairs["Summary"]:
                    dict_value_pairs["Summary"]["Caption"] = []
            dict_value_pairs["Summary"]["Caption"] = ", ".join(sorted(list(caption_set)))

            # Process tag
            caption_set = set()
            for repo_id, key, value in highest_general_keys:
                caption_set.add(key)
                if "Tags" not in dict_value_pairs["Summary"]:
                    dict_value_pairs["Summary"]["Tags"] = []
            dict_value_pairs["Summary"]["Tags"] = ", ".join(sorted(list(caption_set)))

            # Remove old summary

            existing_data.append(dict_value_pairs)

        try:
            with open(json_file_path, 'w') as f:
                json.dump(existing_data, f, indent=4)
            # Get the new file size
            new_size = os.path.getsize(json_file_path)

            if new_size > original_size:
                print(f"JSON data successfully written to {json_file_path} (updated from {original_size} bytes to {new_size} bytes)")
            else:
                print(f"Warning: File size did not increase after writing to {json_file_path} (remained at {original_size} bytes)")

        except IOError as e:
            print(f"Error writing JSON data to {json_file_path}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    else:
        try:
            with open(json_file_path, 'w') as f:
                json.dump([data], f, indent=4)
            print(f"JSON data successfully written to {json_file_path}")
        except IOError as e:
            print(f"Error writing JSON data to {json_file_path}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

def get_json_filepath(image_path: str = ""):

    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Create the json file name with the same name as the image
    json_file_name = f"{image_name}.json"

    # Get the directory of the image
    image_dir = os.path.dirname(image_path)

    # Create the full path for the json file
    json_file_path = os.path.join(image_dir, json_file_name)
    return str(json_file_path)


def print_results(image_path: Path, caption: str, taglist: List[str], ratings: Dict[str, float], character: Dict[str, float], general: Dict[str, float], repo_id: str) -> None:
        print("--------")
        print(f"Caption: {caption}")
        print("--------")
        print(f"Tags: {taglist}")

        print("--------")
        print("Ratings:")
        for k, v in ratings.items():
            print(f"  {k}: {v:.3f}")

        print("--------")
        print(f"Character tags (threshold={opts.char_threshold}):")
        for k, v in character.items():
            print(f"  {k}: {v:.3f}")

        print("--------")
        print(f"General tags (threshold={opts.gen_threshold}):")
        for k, v in general.items():
            print(f"  {k}: {v:.3f}")

@dataclass
class ScriptOptions:
    image_file: list[Path] = field(positional=True, metadata={"nargs": "+"})
    model: str = field(default="vit", metadata={"help": "Model to use. Options: "+str(list(MODEL_REPO_MAP.keys()) + ['all'])})
    gen_threshold: float = field(default=0.35)
    char_threshold: float = field(default=0.75)
    json: bool = field(default=False, metadata={"help": "Will save results to a .json file where the image is and with the same name."})
    print: bool = field(default=False, metadata={"help": "With --json, will print results or else they are muted."})
    summary: bool = field(default=False, metadata={"help": "With --json and --model=all, generate a summary of all best tags ordered by float."})

    def __post_init__(self):
        expanded_files = []
        for file in self.image_file:
            if not file.is_file() and '*' not in str(file):
                raise ValueError("image_file must be a single file or a file glob")
            if '*' in str(file):
                expanded_files.extend(glob.glob(str(file)))
            else:
                expanded_files.append(file)

        if not expanded_files:
            raise ValueError("No files found")

        for file in expanded_files:
            if not Path(file).is_file():
                raise ValueError(f"{file} is not a file")

        self.image_file = expanded_files

        valid_models = list(MODEL_REPO_MAP.keys()) + ['all']
        if ',' in self.model:
            # transform to list
            self.model = self.model.split(',')
            print(self.model)
        for model in self.model:
            if model not in valid_models:
                raise ValueError(f"Invalid model. Must be one of: {valid_models}")

        if 'all' in self.model:
            self.model = list(MODEL_REPO_MAP.keys())


def main(opts: ScriptOptions):
    if opts.model == "all":
        # Use all models, useful to compare model outputs
        tagger_list = list(MODEL_REPO_MAP.keys())
    else :
        if not opts.model == type(list):
            tagger_list = list(opts.model)
        else:
            # it's already a list of models
            tagger_list = opts.model

    model_number = 1
    for tagger_model in tagger_list:
        opts.model = tagger_model
        repo_id = MODEL_REPO_MAP.get(opts.model)

        print(f"Loading model '{opts.model}' from '{repo_id}'...")
        model: nn.Module = timm.create_model("hf-hub:" + str(repo_id)).eval()
        state_dict = timm.models.load_state_dict_from_hf(repo_id)
        model.load_state_dict(state_dict)

        print("Loading tag list...")
        labels: LabelData = load_labels_hf(repo_id=repo_id)

        print("Creating data transform...")
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

        if not opts.image_file :
            print("NO IMAGE LIST !! Should not happen")
            exit(1)

        # process each images with the same model, keeping the model loaded to go faster
        image_nb = 0
        image_total = len(opts.image_file)
        for image_path in opts.image_file:

            print(f"Loading image {image_nb}/{image_total} and preprocessing with {opts.model}")
            # get image
            img_input: Image.Image = Image.open(image_path)
            # ensure image is RGB
            img_input = pil_ensure_rgb(img_input)
            # pad to square with white background
            img_input = pil_pad_square(img_input)
            # run the model's input transform to convert to tensor and rescale
            inputs: Tensor = transform(img_input).unsqueeze(0)
            # NCHW image RGB to BGR
            inputs = inputs[:, [2, 1, 0]]

            print("Running inference...")
            with torch.inference_mode():
                # move model to GPU, if available
                if torch_device.type != "cpu":
                    model = model.to(torch_device)
                    inputs = inputs.to(torch_device)
                # run the model
                outputs = model.forward(inputs)
                # apply the final activation function (timm doesn't support doing this internally)
                outputs = F.sigmoid(outputs)
                # move inputs, outputs, and model back to to cpu if we were on GPU
                if torch_device.type != "cpu":
                    inputs = inputs.to("cpu")
                    outputs = outputs.to("cpu")
                    model = model.to("cpu")

            print("Processing results...")
            caption, taglist, ratings, character, general = get_tags(
                probs=outputs.squeeze(0),
                labels=labels,
                gen_threshold=opts.gen_threshold,
                char_threshold=opts.char_threshold,
            )

            image_nb +=1

            if opts.json:
                # Want summary, it's the last model and there is more than one model
                if opts.summary and len(tagger_list) > 1 and model_number == len(tagger_list):
                    summary=True
                else:
                    summary=False
                write_to_json(image_path, caption, taglist, ratings, character, general, repo_id=repo_id, summary=summary)
                print("Wrote to json file : "+ get_json_filepath(image_path))

            if not opts.json or opts.print:
                print_results(image_path=image_path, caption=caption, taglist=taglist, ratings=ratings, character=character, general=general, repo_id=repo_id)

        # update model number after all images where processed
        model_number += 1

    print("Done!")


if __name__ == "__main__":
    opts, _ = parse_known_args(ScriptOptions)
    # Now handled with @dataclass
    # if opts.model not in MODEL_REPO_MAP or opts.model != "all":
    #     print(f"Available models: {list(MODEL_REPO_MAP.keys())}")
    #     print("You may request for the image to be tested against all models by passing --model=all.")
    #     raise ValueError(f"Unknown model name '{opts.model}'")
    main(opts)
