"""Caption AMASS control renders with a VLM: a description of the rendered
pose plus a generated scene context (clothing, environment, lighting)."""
import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM, BlipForConditionalGeneration, Blip2ForConditionalGeneration, Gemma3ForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from PIL import Image

CAPTION_MODELS = {
    'blip-base': 'Salesforce/blip-image-captioning-base',    # 990MB
    'blip-large': 'Salesforce/blip-image-captioning-large',  # 1.9GB
    'blip2-2.7b': 'Salesforce/blip2-opt-2.7b',               # 15.5GB
    'blip2-flan-t5-xl': 'Salesforce/blip2-flan-t5-xl',       # 15.77GB
    'git-large-coco': 'microsoft/git-large-coco',            # 1.58GB
    'fuse-cap': 'noamrot/FuseCap',                           # 990MB
    'gemma-3-4b-it': 'google/gemma-3-4b-it',
    'qwen': "Qwen/Qwen2.5-VL-7B-Instruct"
}


LIGHTING_SUBCAPTIONS = [
# Natural daylight
"bathed in warm golden hour sunlight",
"lit by soft overcast daylight",
"under harsh midday sun with strong shadows",
"illuminated by early morning light",
"in the blue hour twilight",
"backlit by bright sunlight creating a subtle rim light",
"in dappled sunlight filtering through leaves",
"under the warm glow of late afternoon sun",

# Indoor/artificial
"lit by warm tungsten indoor lighting",
"illuminated by cool fluorescent overhead lights",
"in soft diffused window light from the side",
"under practical room lighting with natural shadows",
"lit by a mix of natural window light and indoor lamps",
"in dim ambient lighting with pools of light",
"illuminated by overhead strip lighting",
"in warm candlelit ambiance",

# Quality/mood
"with soft, diffused lighting that minimizes harsh shadows",
"in dramatic side lighting that emphasizes texture",
"with high contrast lighting creating deep shadows",
"in even, flat lighting typical of overcast conditions",
"with gentle fill light reducing shadow depth",
"in moody low-key lighting",
"under bright, airy lighting",
"with chiaroscuro lighting creating volume",

# Weather-influenced
"in the diffuse light of a cloudy day",
"lit by sunlight breaking through storm clouds",
"in the cool light of a rainy day",
"under the warm light of a hazy summer day",
"in the crisp clear light of winter",
"with soft light filtered through morning mist",
]

CAMERA_SUBCAPTIONS = [
    # Focal length & perspective
    "shot with a 50mm lens at eye level",
    "captured with a 35mm wide angle lens",
    "photographed with an 85mm portrait lens",
    "taken with a 24mm lens showing environmental context",
    "shot with a standard focal length",
    
    # Depth of field
    "with shallow depth of field blurring the background",
    "with deep depth of field keeping foreground and background sharp",
    "bokeh effect in the out-of-focus background",
    "with a slightly defocused background for separation",
    "everything in focus from front to back",
    "with creamy bokeh behind the subject",
    
    # Camera angle
    "from a slightly low angle looking up",
    "from an elevated perspective looking down",
    "at eye level for a natural perspective",
    "from a three-quarter angle",
    "shot from ground level",
    
    # Technical aspects
    "captured with natural grain visible",
    "shot on 35mm film with subtle grain",
    "digital capture with crisp detail",
    "with visible film grain texture",
    "sharp focus on the subject",
    "slightly soft focus for a dreamy quality",
    "high shutter speed freezing motion",
    "with natural chromatic characteristics",
    
    # Photography style
    "in documentary photography style",
    "street photography aesthetic",
    "environmental portrait composition",
    "photojournalistic approach",
    "editorial photography style",
    "lifestyle photography look",
    "candid photography moment",
]

REALISM_SUBCAPTIONS = [
    # Imperfections & authenticity
    "clothing shows natural wrinkles and wear",
    "hair is slightly windblown and imperfect",
    "natural skin texture visible with pores and minor blemishes",
    "fabric appears lived-in with realistic draping",
    "casual, unposed body language",
    "authentic moment captured naturally",
    "clothing fits naturally with realistic folds",
    "hair has natural volume and texture",
    
    # Environmental realism
    "background shows realistic depth and detail",
    "environmental elements appear weathered and authentic",
    "surfaces show realistic texture and wear",
    "natural color grading with accurate tones",
    "realistic ambient occlusion and shadows",
    "practical props and objects visible",
    "authentic materials and textures throughout",
    
    # Atmospheric details
    "atmosphere has subtle haze or particulates",
    "natural color temperature variation across the scene",
    "subtle dust particles visible in light rays",
    "realistic atmospheric perspective with distance",
    "air has visible depth and atmosphere",
    
    # Moment & expression
    "captured in an unguarded moment",
    "natural, relaxed expression",
    "mid-gesture with natural motion blur",
    "between poses in a candid instant",
    "spontaneous, unrehearsed moment",
    "organic body positioning",
    
    # Technical realism
    "no artificial smoothing or beauty filters",
    "natural skin tones without enhancement",
    "realistic color saturation and contrast",
    "subtle lens distortion at edges",
    "natural vignetting from lens characteristics",
    "authentic photographic artifacts",
    "real-world lighting falloff",
    
    # Scene authenticity
    "environment shows signs of use and life",
    "background has realistic clutter and detail",
    "props and objects are contextually appropriate",
    "setting has lived-in, authentic character",
    "scene composition feels unstaged",
]

class ImageCaptioner:
    def __init__(self, model: str = "blip2-2.7b", device: str = "cuda"):
        self.model = model
        self.device = device
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        model_path = CAPTION_MODELS[model]
        if model.startswith('git-'):
            caption_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
        elif model.startswith('blip2-'):
            caption_model = Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=self.dtype)
        elif model.startswith('gemma-'):
            caption_model = Gemma3ForConditionalGeneration.from_pretrained(model_path, device_map="auto")
        elif model.startswith('qwen'):
            caption_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=self.dtype)
        else:
            caption_model = BlipForConditionalGeneration.from_pretrained(model_path, torch_dtype=self.dtype)
        self.caption_processor = AutoProcessor.from_pretrained(model_path)
        self.caption_model = caption_model.eval().to(device)
        self.internal_counter = 0

    def generate_caption(self, image: np.ndarray, index) -> str:
        if self.model.startswith('gemma-') or self.model.startswith('qwen'):
            return self.generate_caption_gemma(image, index)
        else:
            inputs = self.caption_processor(images=image, return_tensors="pt").to(self.device)
            if self.model.startswith('blip2-'):
                inputs = inputs.to(self.dtype)
            tokens = self.caption_model.generate(**inputs, max_new_tokens=100)
            return self.caption_processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()

    def enhance_prompt(self, base_prompt: str, index: int) -> str:
        """
        Enhance the base prompt with randomly sampled lighting, camera, and realism details.
        
        Args:
            base_prompt: The initial prompt generated for the image
            index: Image index for seeding randomization
            
        Returns:
            Enhanced prompt with additional photographic details
        """
        # Seed random for reproducibility per image
        random.seed(index + 42)
        
        # Sample 1 from each category (adjust probabilities as needed)
        lighting = random.choice(LIGHTING_SUBCAPTIONS) if random.random() > 0.1 else ""
        camera = random.choice(CAMERA_SUBCAPTIONS) if random.random() > 0.15 else ""
        realism = random.choice(REALISM_SUBCAPTIONS) if random.random() > 0.1 else ""
        
        # Sometimes sample multiple realism aspects for extra richness
        if random.random() > 0.7:
            realism += ", " + random.choice(REALISM_SUBCAPTIONS)
        
        # Build enhancement clauses
        enhancements = []
        if lighting:
            enhancements.append(lighting)
        if camera:
            enhancements.append(camera)
        if realism:
            enhancements.append(realism)
        
        # Randomly decide whether to add at beginning or end
        if not enhancements:
            return base_prompt
        
        enhancement_text = ", ".join(enhancements)
        
        # 50% chance to add at end, 50% to weave into the middle
        if random.random() > 0.5:
            enhanced = f"{base_prompt.rstrip('.')}; {enhancement_text}."
        else:
            # Insert after first sentence
            sentences = base_prompt.split('. ')
            if len(sentences) > 1:
                enhanced = f"{sentences[0]}; {enhancement_text}. {'. '.join(sentences[1:])}"
            else:
                enhanced = f"{base_prompt.rstrip('.')}; {enhancement_text}."
        
        return enhanced
    
    def generate_caption_gemma(self, image: np.ndarray, index) -> str:
        SCENARIO_GROUPS = {
            "Everyday Indoor": [
                "home kitchen with sunlight through the window",
                "cozy living room with a fireplace",
                "classroom during a lesson",
                "art studio with canvases and paints",
                "modern office workspace with computers",
                "gym or fitness studio",
                "music rehearsal room with instruments",
                "library reading area with tall shelves",
                "workshop with tools and materials",
                "science lab with beakers and microscopes",
                "bedroom with an unmade bed and plants",
                "home office with books and a laptop",
                "corridor lined with lockers",
                "laundry room with clothes drying",
                "craft room with sewing materials",
                "greenhouse filled with potted plants",
                "dining room set for a meal",
                "cozy attic with slanted windows",
                "pantry with shelves of jars",
                "indoor swimming pool area",
                "foyer with a coat rack and umbrella stand",
                "garage with tools and bicycles",
                "children's playroom with toys",
                "basement storage room",
                "indoor climbing gym",
                "museum exhibit hall",
                "aquarium viewing room",
                "recording studio",
                "photography darkroom",
                "lecture hall with rows of seats"
            ],

            "Social & Public": [
                "bustling café with steaming cups on tables",
                "crowded street market with colorful stalls",
                "train or subway platform at rush hour",
                "busy crosswalk in the rain",
                "street festival with hanging lanterns",
                "outdoor concert with a cheering crowd",
                "museum gallery with paintings",
                "airport terminal with people waiting",
                "public library check-out counter",
                "shopping mall atrium",
                "theater lobby with posters",
                "school cafeteria during lunch",
                "city bus interior",
                "hotel lobby with luggage carts",
                "farmers market with fresh produce",
                "marina dock with small boats",
                "amusement park midway",
                "cinema concession stand",
                "bowling alley",
                "university quad with students",
                "sports stadium concourse",
                "street-side food stall",
                "nightclub dance floor",
                "community center gym",
                "roller skating rink",
                "book fair in a tent",
                "outdoor craft fair",
                "public square during a rally",
                "tram stop in a busy district",
                "rooftop bar at sunset"
            ],

            "Nature & Outdoors": [
                "forest trail in autumn",
                "mountain meadow in spring",
                "snow-covered park with benches",
                "beach at sunset",
                "rocky riverside",
                "desert with rolling dunes",
                "lush tropical garden",
                "flower field in bloom",
                "grassy hillside under a blue sky",
                "wetland with tall reeds",
                "pine forest with dappled sunlight",
                "clifftop overlooking the sea",
                "orchard with ripe fruit",
                "countryside path lined with hedges",
                "meadow with grazing animals",
                "rainforest with thick canopy",
                "canyon with layered rock walls",
                "alpine lake with reflections",
                "savanna with acacia trees",
                "coastal bluff with seabirds",
                "volcanic black sand beach",
                "waterfall pool in a jungle",
                "windswept moorland",
                "rock arch along a coast",
                "sandbar in shallow turquoise water",
                "snowy mountain pass",
                "vine-covered ruins in the jungle",
                "wildflower-filled valley",
                "mangrove forest",
                "plateau with panoramic views"
            ],

            "Sports & Action": [
                "soccer field during play",
                "basketball court under bright lights",
                "ice skating rink",
                "yoga class in session",
                "hiking path on a mountain slope",
                "surfing at a beach",
                "track and field stadium",
                "swimming pool deck",
                "tennis court during a match",
                "boxing gym with punching bags",
                "rock climbing wall",
                "dance studio with mirrored walls",
                "volleyball game on sand",
                "ski slope with fresh snow",
                "kayaking on a calm river",
                "equestrian arena with jumps",
                "archery range",
                "indoor trampoline park",
                "table tennis hall",
                "martial arts dojo",
                "rowing team dock",
                "badminton court",
                "fencing salle",
                "cricket pitch",
                "golf course putting green",
                "bouldering gym",
                "roller derby track",
                "snowboard halfpipe"
            ],

            "Rural & Countryside": [
                "vineyard at harvest time",
                "farmyard with animals",
                "barn interior with hay bales",
                "country road with wildflowers",
                "orchard in bloom",
                "windmill on a grassy hill",
                "village square",
                "fishing dock at dawn",
                "field of tall wheat",
                "wooden bridge over a stream",
                "pasture with grazing cows",
                "rustic cabin porch",
                "gravel path lined with fences",
                "old watermill by a river",
                "sheep pasture with rolling hills",
                "market in a rural town",
                "cornfield in late summer",
                "farmhouse kitchen",
                "hayfield during baling",
                "rural bus stop",
                "stone cottage garden",
                "country fairgrounds",
                "apple orchard in autumn",
                "sunflower field",
                "river crossing with stepping stones",
                "farm equipment shed",
                "vine-covered farmhouse",
                "meadow with wild ponies"
            ],

            "Urban & Architectural": [
                "neon-lit alley at night",
                "rooftop garden with city views",
                "open-air food court",
                "plaza with fountains",
                "covered pedestrian bridge",
                "historic cobblestone street",
                "skyscraper observation deck",
                "city park with skyline view",
                "underpass with street art",
                "glass-walled shopping center",
                "public square with a statue",
                "modern tram station",
                "industrial warehouse interior",
                "apartment balcony at dusk",
                "narrow European side street",
                "outdoor staircase in an urban setting",
                "graffiti-covered skate park",
                "parking garage rooftop",
                "metro entrance",
                "canal-side walkway",
                "street café under string lights",
                "ferry terminal",
                "courtyard with potted plants",
                "construction site",
                "hotel rooftop pool",
                "government building steps",
                "art deco cinema exterior"
            ],

            "Unusual / Sci-Fi & Fantasy": [
                "space station corridor",
                "underwater research lab",
                "floating city above clouds",
                "ancient temple ruin",
                "steampunk workshop",
                "enchanted forest with glowing plants",
                "futuristic hover-train platform",
                "ice palace under auroras",
                "alien desert with two suns",
                "crystal cave with shimmering walls",
                "cyberpunk street with neon signs",
                "volcanic landscape with rivers of lava",
                "floating island in the sky",
                "holographic marketplace",
                "time-worn library of magical tomes",
                "subterranean hall with giant roots",
                "ring-shaped orbital habitat",
                "bioluminescent coral canyon",
                "spaceship bridge with holographic displays",
                "ancient stone circle under eclipse",
                "mirror-walled void",
                "labyrinth of shifting walls",
                "planetary surface with aurora-filled sky",
                "hovering crystal monoliths",
                "gilded throne room",
                "starship hangar bay",
                "storm-swept airship deck",
                "undersea domed city"
            ],
        }

        GROUP_ORDER = list(SCENARIO_GROUPS.keys())
        NUM_GROUPS = len(GROUP_ORDER)

        # -----------------------------
        # Pick scenario from index
        # -----------------------------
        def pick_scenario(index: int):
            """Rotate evenly across groups and scenarios within each group."""
            group_idx = index % NUM_GROUPS
            group_name = GROUP_ORDER[group_idx]
            within_idx = (index // NUM_GROUPS) % len(SCENARIO_GROUPS[group_name])
            scenario = SCENARIO_GROUPS[group_name][within_idx]
            return group_name, scenario

        # -----------------------------
        # Build the prompt
        # -----------------------------
        def build_caption_prompt(index: int) -> str:
            group, scenario = pick_scenario(index)
            prompt = (
                f"See the human pose in the image, now imagine clothing for that person with tha pose in a scenario related to {scenario} ({group}).\n"
                "Give me a single, detailed caption that describes the person aspect and their clothing and then the background of the image you have imagined in a coherent couple of sentences. \n"
                "The most important thing is to output a body description that matches the input image as closely as possible, then the background of the image you have imagined.\n"
                "Be concise and precise, don't give me any reasoning, just the output description. 100 tokens max.\n"
            )
            return prompt
        
        text = build_caption_prompt(self.internal_counter)
        text = self.enhance_prompt(text, index)

        messages = [
            {
            "role": "user",
            "content": [
            {"type": "text", "text": text},
            {"type": "image", "image": Image.fromarray(image)}
            ]
            }
        ]
        inputs = self.caption_processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        tokens = self.caption_model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.8, top_p=0.9, pad_token_id=self.caption_processor.tokenizer.eos_token_id)
        tokens = tokens[0][input_len:]
        self.internal_counter += 1
        return self.caption_processor.decode(tokens, skip_special_tokens=True).strip()
    
    def pil_resize_image(self, image):
        width, height = image.size
        new_height = 256
        new_width = int(width * new_height / height)
        return image.resize((new_width, new_height))


if __name__ == "__main__":
    import os
    from os import environ
    import matplotlib.pyplot as plt
    import json
    import random
    from tqdm import tqdm

    MAX_TASKS = 4

    def _get_start_end_index(images):
        if "SLURM_ARRAY_TASK_ID" not in environ:
            return 0, len(images)
        task_id = int(environ["SLURM_ARRAY_TASK_ID"])
        num_in_one_bucket = len(images) // MAX_TASKS
        return task_id * num_in_one_bucket, min(len(images), (task_id + 1) * num_in_one_bucket)

    import argparse
    parser = argparse.ArgumentParser(description="Caption control renders with a VLM")
    parser.add_argument("--images_folder", required=True, help="Folder of control render PNGs to caption")
    parser.add_argument("--out_dir", required=True, help="Output folder for per-image caption JSONs")
    parser.add_argument("--model", default="gemma-3-4b-it")
    cli_args = parser.parse_args()

    image_captioner = ImageCaptioner(cli_args.model, device="cuda")
    images_folder = cli_args.images_folder
    out_dir = cli_args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    filenames = sorted(os.listdir(images_folder))
    start, end = _get_start_end_index(filenames)
    print(f"Reading subset from {start} to {end}, total: {len(filenames)}")
    indices = list(range(start, end))
    random.shuffle(indices)
    for index in tqdm(indices):
        image_file = filenames[index]
        image_path = os.path.join(images_folder, image_file)
        image = Image.open(image_path).convert("RGB")
        out_json = os.path.join(out_dir, image_file.replace(".png", ".json"))
        if os.path.exists(out_json):
            print(f"Skipping {image_file}, out json already exists")
            continue
        caption = image_captioner.generate_caption(np.array(image), index)
        data = {
            "caption": caption
        }
        with open(out_json, 'w') as f:
            json.dump(data, f)
        print(f"Caption for {image_file}: {caption}")
