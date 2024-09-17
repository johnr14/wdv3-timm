# wdv3-timm

small example thing showing how to use `timm` to run the WD Tagger V3 models.

## How To Use

1. clone the repository and enter the directory:
```sh
git clone https://github.com/neggles/wdv3-timm.git
cd wd3-timm
```

2. Create a virtual environment and install the Python requirements.

If you're using Linux, you can use the provided script:
```sh
bash setup.sh
```

Or if you're on Windows (or just want to do it manually), you can do the following:
```sh
# Create virtual environment
python3.10 -m venv .venv
# Activate it
source .venv/bin/activate
# Upgrade pip/setuptools/wheel
python -m pip install -U pip setuptools wheel
# At this point, optionally you can install PyTorch manually (e.g. if you are not using an nVidia GPU)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Install requirements
python -m pip install -r requirements.txt
```

3. Run the example script, picking one of the 3 models to use or all of them:
```sh
python wdv3_timm.py --model=<swinv2|convnext|vit|all> path/to/image.png
```

Example output from `python wdv3_timm.py --model=vit a_picture_of_ganyu.png`:
```sh
Loading model 'vit' from 'SmilingWolf/wd-vit-tagger-v3'...
Loading tag list...
Creating data transform...
Loading image and preprocessing...
Running inference...
Processing results...
--------
Caption: 1girl, horns, solo, bell, ahoge, colored_skin, blue_skin, neck_bell, looking_at_viewer, purple_eyes, upper_body, blonde_hair, long_hair, goat_horns, blue_hair, off_shoulder, sidelocks, bare_shoulders, alternate_costume, shirt, black_shirt, cowbell, ganyu_(genshin_impact)
--------
Tags: 1girl, horns, solo, bell, ahoge, colored skin, blue skin, neck bell, looking at viewer, purple eyes, upper body, blonde hair, long hair, goat horns, blue hair, off shoulder, sidelocks, bare shoulders, alternate costume, shirt, black shirt, cowbell, ganyu \(genshin impact\)
--------
Ratings:
  general: 0.827
  sensitive: 0.199
  questionable: 0.001
  explicit: 0.001
--------
Character tags (threshold=0.75):
  ganyu_(genshin_impact): 0.991
--------
General tags (threshold=0.35):
  1girl: 0.996
  horns: 0.950
  solo: 0.947
  bell: 0.918
  ahoge: 0.897
  colored_skin: 0.881
  blue_skin: 0.872
  neck_bell: 0.854
  looking_at_viewer: 0.817
  purple_eyes: 0.734
  upper_body: 0.615
  blonde_hair: 0.609
  long_hair: 0.607
  goat_horns: 0.524
  blue_hair: 0.496
  off_shoulder: 0.472
  sidelocks: 0.470
  bare_shoulders: 0.464
  alternate_costume: 0.437
  shirt: 0.427
  black_shirt: 0.417
  cowbell: 0.415
```

## Do more with batch mode !
Files will be processed without unloading the model, saving lots of time.

It's possible to have multiple or all model be used one after the other.
It is done by unloading and loading a new model after all the files where done with the previous model.

```sh
# with file list you provide
python wdv3_timm.py --model=all path/to/image_1.png path/to/image_2.png path/to/image_3.png
# with file list created from shell
python wdv3_timm.py --model=all path/to/image_*.png
# or with glob that are handled by python, put in quotes
python wdv3_timm.py --model=all "path/*/image_*.png"
# only with selected models :
python wdv3_timm.py --model=vit,convnext "path/*/image_*.png"

```
##  Save to JSON !
Save output to JSON in the same path that the file is with the same filename but with a .json extension.
`--json` will disable printing the results, if you want them back, use `--print`.
If a JSON is already existant, it will append to it.

```sh
# will create a filepath/filename.json file
python wdv3_timm.py --model=all path/to/image_*.png --json
# or to also print the results
python wdv3_timm.py --model=all path/to/image_*.png --json --print
```
The JSON will look like :

<details>
  <summary>Click me to view JSON output for all 3 models !</summary>

```JSON
[
    {
        "Caption": "1girl, solo, horns, blue_hair, long_hair, detached_sleeves, ahoge, gloves, purple_eyes, bell, orb, breasts, looking_at_viewer, black_gloves, white_background, bare_shoulders, smile, medium_breasts, neck_bell, thighlet, sidelocks, vision_(genshin_impact), standing, white_sleeves, tassel, gold_trim, simple_background, bow, pantyhose, waist_cape, flower_knot, bodystocking, feet_out_of_frame, low_ponytail, chinese_knot, long_sleeves, ganyu_(genshin_impact)",
        "Tags": "1girl, solo, horns, blue hair, long hair, detached sleeves, ahoge, gloves, purple eyes, bell, orb, breasts, looking at viewer, black gloves, white background, bare shoulders, smile, medium breasts, neck bell, thighlet, sidelocks, vision \\(genshin impact\\), standing, white sleeves, tassel, gold trim, simple background, bow, pantyhose, waist cape, flower knot, bodystocking, feet out of frame, low ponytail, chinese knot, long sleeves, ganyu \\(genshin impact\\)",
        "Ratings": {
            "general": "0.17550515",
            "sensitive": "0.83533895",
            "questionable": "0.0011807431",
            "explicit": "0.00017911881"
        },
        "Character": {
            "ganyu_(genshin_impact)": "0.9917346"
        },
        "General": {
            "1girl": "0.99861884",
            "solo": "0.9774026",
            "horns": "0.9733339",
            "blue_hair": "0.9678151",
            "long_hair": "0.96172965",
            "detached_sleeves": "0.9510404",
            "ahoge": "0.94540447",
            "gloves": "0.9342438",
            "purple_eyes": "0.9304822",
            "bell": "0.92110497",
            "orb": "0.9137795",
            "breasts": "0.9032445",
            "looking_at_viewer": "0.90013504",
            "black_gloves": "0.862591",
            "white_background": "0.8615776",
            "bare_shoulders": "0.8545493",
            "smile": "0.82447666",
            "medium_breasts": "0.8215981",
            "neck_bell": "0.79515177",
            "thighlet": "0.7304673",
            "sidelocks": "0.7278657",
            "vision_(genshin_impact)": "0.6980954",
            "standing": "0.690362",
            "white_sleeves": "0.6442809",
            "tassel": "0.6387444",
            "gold_trim": "0.612045",
            "simple_background": "0.5703776",
            "bow": "0.49510783",
            "pantyhose": "0.46065962",
            "waist_cape": "0.44340992",
            "flower_knot": "0.4369225",
            "bodystocking": "0.43336046",
            "feet_out_of_frame": "0.42403945",
            "low_ponytail": "0.40857768",
            "chinese_knot": "0.4051522",
            "long_sleeves": "0.36612618"
        },
        "Repo_id": "SmilingWolf/wd-vit-tagger-v3"
    },
    {
        "Caption": "1girl, solo, horns, long_hair, blue_hair, detached_sleeves, gloves, ahoge, breasts, bell, looking_at_viewer, purple_eyes, orb, black_gloves, smile, bare_shoulders, medium_breasts, white_background, vision_(genshin_impact), neck_bell, gold_trim, white_sleeves, sidelocks, thighlet, standing, tassel, simple_background, pantyhose, bodystocking, low_ponytail, flower_knot, chinese_knot, long_sleeves, feet_out_of_frame, ganyu_(genshin_impact)",
        "Tags": "1girl, solo, horns, long hair, blue hair, detached sleeves, gloves, ahoge, breasts, bell, looking at viewer, purple eyes, orb, black gloves, smile, bare shoulders, medium breasts, white background, vision \\(genshin impact\\), neck bell, gold trim, white sleeves, sidelocks, thighlet, standing, tassel, simple background, pantyhose, bodystocking, low ponytail, flower knot, chinese knot, long sleeves, feet out of frame, ganyu \\(genshin impact\\)",
        "Ratings": {
            "general": "0.09444821",
            "sensitive": "0.9110725",
            "questionable": "0.001273665",
            "explicit": "0.00019249455"
        },
        "Character": {
            "ganyu_(genshin_impact)": "0.989583"
        },
        "General": {
            "1girl": "0.9990073",
            "solo": "0.986086",
            "horns": "0.97691685",
            "long_hair": "0.976655",
            "blue_hair": "0.97018594",
            "detached_sleeves": "0.9630663",
            "gloves": "0.9528584",
            "ahoge": "0.9466164",
            "breasts": "0.939335",
            "bell": "0.9139086",
            "looking_at_viewer": "0.90088356",
            "purple_eyes": "0.9004356",
            "orb": "0.89351803",
            "black_gloves": "0.8897617",
            "smile": "0.8828965",
            "bare_shoulders": "0.87559116",
            "medium_breasts": "0.8694743",
            "white_background": "0.8646937",
            "vision_(genshin_impact)": "0.7989824",
            "neck_bell": "0.7968426",
            "gold_trim": "0.77870077",
            "white_sleeves": "0.74657595",
            "sidelocks": "0.7191376",
            "thighlet": "0.7173794",
            "standing": "0.6971251",
            "tassel": "0.6534954",
            "simple_background": "0.6306383",
            "pantyhose": "0.60918325",
            "bodystocking": "0.5880574",
            "low_ponytail": "0.58335847",
            "flower_knot": "0.55061",
            "chinese_knot": "0.4911994",
            "long_sleeves": "0.40371013",
            "feet_out_of_frame": "0.36766034"
        },
        "Repo_id": "SmilingWolf/wd-swinv2-tagger-v3"
    },
    {
        "Caption": "1girl, solo, long_hair, ahoge, horns, blue_hair, orb, detached_sleeves, gloves, bell, purple_eyes, breasts, looking_at_viewer, bare_shoulders, white_background, black_gloves, medium_breasts, neck_bell, vision_(genshin_impact), tassel, smile, white_sleeves, gold_trim, sidelocks, simple_background, standing, thighlet, bodystocking, pantyhose, flower_knot, chinese_knot, low_ponytail, bow, waist_cape, feet_out_of_frame, ganyu_(genshin_impact)",
        "Tags": "1girl, solo, long hair, ahoge, horns, blue hair, orb, detached sleeves, gloves, bell, purple eyes, breasts, looking at viewer, bare shoulders, white background, black gloves, medium breasts, neck bell, vision \\(genshin impact\\), tassel, smile, white sleeves, gold trim, sidelocks, simple background, standing, thighlet, bodystocking, pantyhose, flower knot, chinese knot, low ponytail, bow, waist cape, feet out of frame, ganyu \\(genshin impact\\)",
        "Ratings": {
            "general": "0.11932882",
            "sensitive": "0.87867385",
            "questionable": "0.0012409396",
            "explicit": "0.00015736303"
        },
        "Character": {
            "ganyu_(genshin_impact)": "0.9868869"
        },
        "General": {
            "1girl": "0.99817467",
            "solo": "0.99103117",
            "long_hair": "0.9574774",
            "ahoge": "0.957307",
            "horns": "0.9566982",
            "blue_hair": "0.953568",
            "orb": "0.9492937",
            "detached_sleeves": "0.94679636",
            "gloves": "0.946109",
            "bell": "0.93114275",
            "purple_eyes": "0.9196103",
            "breasts": "0.9013523",
            "looking_at_viewer": "0.8901093",
            "bare_shoulders": "0.8684672",
            "white_background": "0.8641212",
            "black_gloves": "0.8531206",
            "medium_breasts": "0.8330931",
            "neck_bell": "0.7989295",
            "vision_(genshin_impact)": "0.793058",
            "tassel": "0.78462917",
            "smile": "0.77972054",
            "white_sleeves": "0.72669303",
            "gold_trim": "0.68204206",
            "sidelocks": "0.6696157",
            "simple_background": "0.64645475",
            "standing": "0.6338763",
            "thighlet": "0.6263255",
            "bodystocking": "0.6048315",
            "pantyhose": "0.53592044",
            "flower_knot": "0.48086384",
            "chinese_knot": "0.4745855",
            "low_ponytail": "0.4641142",
            "bow": "0.4116591",
            "waist_cape": "0.38854668",
            "feet_out_of_frame": "0.35137108"
        },
        "Repo_id": "SmilingWolf/wd-convnext-tagger-v3"
    }
]

```

</details>


### JSON summary
Do you want to keep the highest tags from all different models ?
Here you can, but it will ignore cutoff.
Only works with `--json`

Will also tell what models made the highest certainty for selected tags.
For more control, you will have to write your own JSON parser.

```sh
# will create a filepath/filename.json file and summary
python wdv3_timm.py --model=all path/to/image_*.png --json --summary
```
The JSON summary will look like :

<details>
  <summary>Click me to view JSON summary !</summary>


```JSON

 {
        "Summary": {
            "General": {
                "1girl": "0.9990073",
                "solo": "0.99103117",
                "horns": "0.97691685",
                "blue_hair": "0.97018594",
                "long_hair": "0.976655",
                "detached_sleeves": "0.9630663",
                "ahoge": "0.95730704",
                "gloves": "0.95285827",
                "purple_eyes": "0.9304822",
                "bell": "0.9311429",
                "orb": "0.94929373",
                "breasts": "0.939335",
                "looking_at_viewer": "0.90088356",
                "black_gloves": "0.88976157",
                "white_background": "0.8646937",
                "bare_shoulders": "0.87559104",
                "smile": "0.8828964",
                "medium_breasts": "0.8694741",
                "neck_bell": "0.79893",
                "thighlet": "0.7304673",
                "sidelocks": "0.7278657",
                "vision_(genshin_impact)": "0.79898244",
                "standing": "0.69712484",
                "white_sleeves": "0.74657583",
                "tassel": "0.7846296",
                "gold_trim": "0.77870077",
                "simple_background": "0.64645475",
                "bow": "0.49510783",
                "pantyhose": "0.60918343",
                "waist_cape": "0.44340992",
                "flower_knot": "0.55061",
                "bodystocking": "0.6048321",
                "feet_out_of_frame": "0.42403945",
                "low_ponytail": "0.5833588",
                "chinese_knot": "0.49119928",
                "long_sleeves": "0.4037104"
            },
            "General_Models": {
                "SmilingWolf/wd-swinv2-tagger-v3": [
                    "1girl",
                    "horns",
                    "blue_hair",
                    "long_hair",
                    "detached_sleeves",
                    "gloves",
                    "breasts",
                    "looking_at_viewer",
                    "black_gloves",
                    "white_background",
                    "bare_shoulders",
                    "smile",
                    "medium_breasts",
                    "vision_(genshin_impact)",
                    "standing",
                    "white_sleeves",
                    "gold_trim",
                    "pantyhose",
                    "flower_knot",
                    "low_ponytail",
                    "chinese_knot",
                    "long_sleeves"
                ],
                "SmilingWolf/wd-convnext-tagger-v3": [
                    "solo",
                    "ahoge",
                    "bell",
                    "orb",
                    "neck_bell",
                    "tassel",
                    "simple_background",
                    "bodystocking"
                ],
                "SmilingWolf/wd-vit-tagger-v3": [
                    "purple_eyes",
                    "thighlet",
                    "sidelocks",
                    "bow",
                    "waist_cape",
                    "feet_out_of_frame"
                ]
            },
            "Ratings": {
                "general": "0.17550515",
                "sensitive": "0.9110725",
                "questionable": "0.0012736674",
                "explicit": "0.00019249438"
            },
            "Ratings_Models": {
                "SmilingWolf/wd-vit-tagger-v3": [
                    "general"
                ],
                "SmilingWolf/wd-swinv2-tagger-v3": [
                    "sensitive",
                    "questionable",
                    "explicit"
                ]
            },
            "Character": {
                "ganyu_(genshin_impact)": "0.9917346"
            },
            "Character_Models": {
                "SmilingWolf/wd-vit-tagger-v3": [
                    "ganyu_(genshin_impact)"
                ]
            },
            "Caption": "1girl, ahoge, bare_shoulders, bell, black_gloves, blue_hair, bodystocking, bow, breasts, chinese_knot, detached_sleeves, feet_out_of_frame, flower_knot, gloves, gold_trim, horns, long_hair, long_sleeves, looking_at_viewer, low_ponytail, medium_breasts, neck_bell, orb, pantyhose, purple_eyes, sidelocks, simple_background, smile, solo, standing, tassel, thighlet, vision_(genshin_impact), waist_cape, white_background, white_sleeves",
            "Tags": "1girl, ahoge, bare_shoulders, bell, black_gloves, blue_hair, bodystocking, bow, breasts, chinese_knot, detached_sleeves, feet_out_of_frame, flower_knot, gloves, gold_trim, horns, long_hair, long_sleeves, looking_at_viewer, low_ponytail, medium_breasts, neck_bell, orb, pantyhose, purple_eyes, sidelocks, simple_background, smile, solo, standing, tassel, thighlet, vision_(genshin_impact), waist_cape, white_background, white_sleeves"
        }
    }
```
### JSON tag description
If you want to have tag descriptions in your JSON to add context to a Large Language Model, you can now !
But be warned, the descriptions have messy and not uniform format. Best would be to keep only first line.
Also, will cache the tags to `danbooru_tags_desc.json` and will fetch any new tag from [Danbooru](https://danbooru.donmai.us).
```sh
# will add summary and tag descriptions
python wdv3_timm.py --model=all path/to/image_*.png --json --summary --description
# or just tag descriptions
python wdv3_timm.py --model=all path/to/image_*.png --json --description
```
