import os
import torch
import fire
from PIL import Image
from pathlib import Path

# EasyControl imports
from EasyControl.train.src.pipeline import FluxPipeline
from EasyControl.train.src.transformer_flux import FluxTransformer2DModel
from EasyControl.train.src.lora_helper import set_single_lora


CAPTIONS = [
   "A 25-year-old Black woman in bright pink running leggings and white sneakers sprinting down a busy New York sidewalk at dawn, steam rising from manholes, 4K ultra-realistic photography, full portrait composition, no borders, professional sports photography lighting",
   "An elderly Japanese man in traditional running shorts jogging through Tokyo's neon-lit streets at night, reflections on wet pavement, ultra-high definition, photorealistic detail, complete figure in frame, cinematic quality",
   "A teenage Latina girl in school track uniform running past colorful street murals in East LA, backpack bouncing, 8K resolution, hyperrealistic rendering, full body portrait, studio-quality lighting, no black borders",
   "A middle-aged white businessman in a loosened tie and dress shoes running to catch a bus in London's financial district, ultra-realistic 4K photography, complete portrait framing, professional photojournalism style",
   "A young South Asian woman in hijab and modest athletic wear jogging through a bustling Mumbai marketplace, photorealistic quality, 4K ultra-high definition, full portrait composition, vibrant color grading",
   "A 30-something Native American man in traditional ribbon shirt running during a community marathon on a reservation, ultra-realistic photography, 4K resolution, complete figure portrait, natural lighting, no borders",
   "A college student in university sweatshirt sprinting across campus quad with autumn leaves swirling around, hyperrealistic 8K quality, full body in frame, cinematic depth of field, professional sports photography",
   "A fit woman in her 40s trail running through misty Pacific Northwest forest, Douglas firs towering overhead, ultra-realistic 4K photography, complete portrait composition, natural outdoor lighting, no black borders",
   "A sunburned surfer in board shorts running along Bondi Beach at sunset, waves crashing nearby, photorealistic quality, 4K ultra-high definition, full figure portrait, golden hour lighting",
   "An Inuit teenager racing across Arctic tundra in thermal gear, Northern Lights dancing above, ultra-realistic photography, 8K resolution, complete body composition, dramatic natural lighting, no borders",
   "A Masai warrior in traditional attire running across African savanna with acacia trees silhouetted against orange sky, hyperrealistic 4K quality, full portrait framing, cinematic lighting",
   "A Brazilian woman jogging along Copacabana beach in carnival-colored bikini and running shoes, ultra-realistic photography, 4K resolution, complete figure in frame, tropical lighting, no black borders",
   "An Australian Aboriginal man in modern athletic wear running through red desert landscape near Uluru, photorealistic 8K quality, full body portrait, dramatic outback lighting",
   "A Scandinavian woman in bright winter gear running through snow-covered pine forest, breath visible in cold air, ultra-realistic 4K photography, complete portrait composition, winter natural lighting, no borders",
   "A high school cross-country runner in team colors racing through suburban neighborhood course, hyperrealistic quality, 4K ultra-high definition, full figure portrait, athletic photography style",
   "A triathlete in wetsuit and running shoes transitioning from swim to run portion of race, ultra-realistic photography, 8K resolution, complete body composition, professional sports lighting, no black borders",
   "A marathon runner hitting the wall at mile 20, grimacing but determined, crowd cheering, photorealistic 4K quality, full portrait framing, dynamic sports photography",
   "A military recruit in camouflage running through obstacle course, drill sergeant shouting encouragement, ultra-realistic photography, 4K resolution, complete figure in frame, military training lighting, no borders",
   "A boxing trainer in gym clothes doing roadwork through industrial district at 5 AM, hyperrealistic 8K quality, full body portrait, early morning urban lighting",
   "A determined runner in rain gear splashing through puddles during heavy downpour, streetlights reflecting, ultra-realistic 4K photography, complete portrait composition, dramatic weather lighting, no black borders",
   "A desert runner in protective clothing and hydration pack racing across sand dunes in blazing heat, photorealistic quality, 4K ultra-high definition, full figure portrait, harsh desert lighting",
   "A winter jogger in multiple layers running through snowy city park, snow falling steadily, ultra-realistic photography, 8K resolution, complete body composition, winter atmosphere lighting, no borders",
   "A tropical runner in minimal clothing sprinting through humid jungle path, exotic birds overhead, hyperrealistic 4K quality, full portrait framing, dappled jungle lighting",
   "A mountain runner at high altitude, breathing hard in thin air with peaks in background, ultra-realistic photography, 4K resolution, complete figure in frame, alpine lighting, no black borders",
   "A 7-year-old child in superhero costume flying across playground, cape streaming behind, photorealistic 8K quality, full body portrait, playful outdoor lighting",
   "A 70-year-old grandmother in pastel tracksuit jogging through retirement community, neighbors waving, ultra-realistic 4K photography, complete portrait composition, warm community lighting, no borders",
   "A teenage boy in oversized basketball jersey running late to school, backpack bouncing, hyperrealistic quality, 4K ultra-high definition, full figure portrait, morning sunlight",
   "A middle-aged dad in cargo shorts chasing after kids in suburban park, ultra-realistic photography, 8K resolution, complete body composition, family-friendly lighting, no black borders",
   "A 5-year-old girl in tutu and sneakers running through sprinklers on summer day, photorealistic 4K quality, full portrait framing, bright summer lighting",
   "A nurse in scrubs rushing between hospital wings during emergency shift change, ultra-realistic photography, 4K resolution, complete figure in frame, hospital fluorescent lighting, no borders",
   "A postal worker in uniform running to deliver urgent package before deadline, hyperrealistic 8K quality, full body portrait, urban daylight photography",
   "A lifeguard in red swimsuit sprinting along beach toward swimmers in distress, ultra-realistic 4K photography, complete portrait composition, beach rescue lighting, no black borders",
   "A firefighter in partial gear running toward emergency scene, helmet in hand, photorealistic quality, 4K ultra-high definition, full figure portrait, emergency scene lighting",
   "A pizza delivery person in company shirt running up apartment stairs with hot pizza, ultra-realistic photography, 8K resolution, complete body composition, indoor stairwell lighting, no borders",
   "A Scottish Highland Games competitor in kilt running between events, bagpipes playing nearby, hyperrealistic 4K quality, full portrait framing, Highland festival lighting",
   "A Mexican woman in embroidered huipil running through Day of the Dead festival, ultra-realistic photography, 4K resolution, complete figure in frame, festival celebration lighting, no black borders",
   "A Bollywood dancer in flowing sari running through wedding celebration scene, photorealistic 8K quality, full body portrait, colorful Indian wedding lighting",
   "A Greek runner recreating ancient Olympic tradition on original marathon route, ultra-realistic 4K photography, complete portrait composition, Mediterranean sunlight, no borders",
   "An Irish step dancer in competition dress running between performance venues, hyperrealistic quality, 4K ultra-high definition, full figure portrait, stage lighting effects",
   "A punk rocker with mohawk and studded jacket running from overzealous security guard, ultra-realistic photography, 8K resolution, complete body composition, urban concert venue lighting, no black borders",
   "A hip-hop dancer in streetwear running to underground dance battle in subway station, photorealistic 4K quality, full portrait framing, subway tunnel lighting",
   "A goth teenager in all black running through cemetery at midnight, full moon overhead, ultra-realistic photography, 4K resolution, complete figure in frame, moonlight illumination, no borders",
   "A cosplayer in elaborate anime costume running to convention center, prop sword attached, hyperrealistic 8K quality, full body portrait, convention center lighting",
   "A street artist with paint-stained clothes running from pursuing police officer, ultra-realistic 4K photography, complete portrait composition, urban street lighting, no black borders",
   "A tourist in safari gear running alongside wildebeest migration in Serengeti, photorealistic quality, 4K ultra-high definition, full figure portrait, African savanna lighting",
   "A local guide in traditional dress running up Machu Picchu steps with tour group following, ultra-realistic photography, 8K resolution, complete body composition, Andean mountain lighting, no borders",
   "A pilgrim on Camino de Santiago running final stretch to cathedral in Santiago, hyperrealistic 4K quality, full portrait framing, Spanish countryside lighting",
   "A marathon participant running across Golden Gate Bridge in San Francisco fog, ultra-realistic photography, 4K resolution, complete figure in frame, foggy bridge atmosphere, no black borders",
   "A jogger running past cherry blossoms in full bloom in Kyoto temple grounds, photorealistic 8K quality, full body portrait, Japanese temple garden lighting",
   "A zombie apocalypse survivor in torn clothes running through abandoned city street, ultra-realistic 4K photography, complete portrait composition, post-apocalyptic lighting, no borders",
   "A time traveler in Victorian dress running through modern cityscape looking confused, hyperrealistic quality, 4K ultra-high definition, full figure portrait, urban contrast lighting",
   "A superhero in training running across rooftops with city skyline backdrop, ultra-realistic photography, 8K resolution, complete body composition, dramatic rooftop lighting, no black borders",
   "A space colonist in futuristic gear running through Mars habitat dome, photorealistic 4K quality, full portrait framing, sci-fi atmospheric lighting",
   "A medieval peasant in rough tunic running through castle courtyard during siege, ultra-realistic photography, 4K resolution, complete figure in frame, medieval fortress lighting, no borders",
   "A plus-size woman confidently running 5K in body-positive athletic wear, smile beaming, hyperrealistic 8K quality, full body portrait, inspirational outdoor lighting",
   "A tall basketball player in team uniform running fast break down court, ultra-realistic 4K photography, complete portrait composition, basketball arena lighting, no black borders",
   "A petite gymnast in leotard running across tumbling floor for routine, photorealistic quality, 4K ultra-high definition, full figure portrait, gymnastics competition lighting",
   "A muscular bodybuilder in tank top doing cardio run through Venice Beach, ultra-realistic photography, 8K resolution, complete body composition, California beach lighting, no borders",
   "A lean distance runner maintaining marathon pace through city streets, hyperrealistic 4K quality, full portrait framing, urban marathon lighting",
   "A fashion model in haute couture gown and heels running down Parisian runway, ultra-realistic photography, 4K resolution, complete figure in frame, fashion show lighting, no black borders",
   "A 1920s flapper in beaded dress and T-bar shoes running through speakeasy, photorealistic 8K quality, full body portrait, vintage jazz club lighting",
   "A 1980s fitness enthusiast in neon leotard and leg warmers jogging through mall, ultra-realistic 4K photography, complete portrait composition, retro mall lighting, no borders",
   "A steampunk inventor in goggles and gears running through industrial landscape, hyperrealistic quality, 4K ultra-high definition, full figure portrait, Victorian industrial lighting",
   "A minimalist runner in simple white outfit running through zen garden, ultra-realistic photography, 8K resolution, complete body composition, serene garden lighting, no black borders",
   "A new mother in athletic wear jogging through neighborhood park at sunrise, photorealistic 4K quality, full portrait framing, golden morning light",
   "A heartbroken teenager running away from home with backpack, tears streaming, ultra-realistic photography, 4K resolution, complete figure in frame, emotional dramatic lighting, no borders",
   "A lottery winner running to bank in pajamas clutching winning ticket, pure joy, hyperrealistic 8K quality, full body portrait, excited celebration lighting",
   "A guilty dog owner chasing escaped pet through neighborhood, leash in hand, ultra-realistic 4K photography, complete portrait composition, suburban neighborhood lighting, no black borders",
   "A bride in running shoes under wedding dress racing to ceremony, late but laughing, photorealistic quality, 4K ultra-high definition, full figure portrait, wedding day lighting",
   "A polar explorer in arctic gear running to shelter as blizzard approaches, ultra-realistic photography, 8K resolution, complete body composition, harsh arctic lighting, no borders",
   "A desert nomad in flowing robes running toward oasis mirage in scorching heat, hyperrealistic 4K quality, full portrait framing, intense desert sun lighting",
   "A storm chaser in protective gear running toward tornado funnel for research, ultra-realistic photography, 4K resolution, complete figure in frame, dramatic storm lighting, no black borders",
   "A tropical islander running from approaching tsunami wave, palm trees bending, photorealistic 8K quality, full body portrait, disaster emergency lighting",
   "A mountain climber running from avalanche down steep alpine slope, ultra-realistic 4K photography, complete portrait composition, alpine emergency lighting, no borders",
   "A cyberpunk runner with LED implants sprinting through neon-drenched mega-city, hyperrealistic quality, 4K ultra-high definition, full figure portrait, cyberpunk neon lighting",
   "A bio-enhanced human with visible mechanical augmentations running at superhuman speed, ultra-realistic photography, 8K resolution, complete body composition, sci-fi enhancement lighting, no black borders",
   "A retro-futuristic runner in chrome suit racing through 1950s vision of 2020, photorealistic 4K quality, full portrait framing, retro-futuristic lighting",
   "A digital avatar runner moving through pixelated video game landscape, ultra-realistic photography, 4K resolution, complete figure in frame, digital world lighting, no borders",
   "A virtual reality athlete competing in simulated marathon environment, hyperrealistic 8K quality, full body portrait, VR simulation lighting",
   "A pack runner keeping pace with wolves through wilderness preserve, ultra-realistic 4K photography, complete portrait composition, wild nature lighting, no black borders",
   "A beach jogger racing dolphins swimming parallel in surf, photorealistic quality, 4K ultra-high definition, full figure portrait, ocean beach lighting",
   "A forest runner followed by curious deer family through morning mist, ultra-realistic photography, 8K resolution, complete body composition, misty forest lighting, no borders",
   "A canyon runner echoing footsteps off red rock walls while eagles soar, hyperrealistic 4K quality, full portrait framing, desert canyon lighting",
   "A prairie runner disturbing grasshoppers that create cloud around them, ultra-realistic photography, 4K resolution, complete figure in frame, prairie grassland lighting, no black borders",
   "A Roman messenger in toga running through Forum with urgent scroll, photorealistic 8K quality, full body portrait, ancient Roman architectural lighting",
   "A Wild West outlaw in chaps and spurs running from sheriff's posse, ultra-realistic 4K photography, complete portrait composition, Old West desert lighting, no borders",
   "A World War II resistance fighter running secret message through occupied city, hyperrealistic quality, 4K ultra-high definition, full figure portrait, wartime urban lighting",
   "A Colonial American patriot running to spread news of revolution, ultra-realistic photography, 8K resolution, complete body composition, colonial village lighting, no black borders",
   "A gold rush prospector running toward newly discovered claim site, photorealistic 4K quality, full portrait framing, California gold country lighting",
   "A street performer in mime costume running invisible marathon for crowd tips, ultra-realistic photography, 4K resolution, complete figure in frame, street performance lighting, no borders",
   "A contemporary dancer interpreting flight through urban environment installation, hyperrealistic 8K quality, full body portrait, artistic installation lighting",
   "A photographer running to capture perfect golden hour shot, cameras bouncing, ultra-realistic 4K photography, complete portrait composition, golden hour lighting, no black borders",
   "A graffiti artist in spray paint-stained clothes running from security with artwork, photorealistic quality, 4K ultra-high definition, full figure portrait, urban alley lighting",
   "A performance artist in white body paint running through art gallery opening, ultra-realistic photography, 8K resolution, complete body composition, gallery exhibition lighting, no borders",
   "A neighborhood watch volunteer running to help elderly resident with groceries, hyperrealistic 4K quality, full portrait framing, community neighborhood lighting",
   "A community organizer racing to rally before city council meeting, ultra-realistic photography, 4K resolution, complete figure in frame, civic center lighting, no black borders",
   "A volunteer coach running alongside youth athlete offering encouragement, photorealistic 8K quality, full body portrait, community sports field lighting",
   "A parent running to school pickup line realizing they're late again, ultra-realistic 4K photography, complete portrait composition, suburban school zone lighting, no borders",
   "A good Samaritan running to help accident victim while calling emergency services, hyperrealistic quality, 4K ultra-high definition, full figure portrait, emergency scene lighting",
   "A parkour athlete in urban gear running across rooftop obstacle course, ultra-realistic photography, 8K resolution, complete body composition, urban parkour lighting, no black borders",
   "An escaped prisoner in orange jumpsuit running through cornfield at night, photorealistic 4K quality, full portrait framing, moonlit cornfield lighting",
   "A Secret Service agent in dark suit running to protect VIP in crowd, ultra-realistic photography, 4K resolution, complete figure in frame, security detail lighting, no borders",
   "A paparazzi photographer chasing celebrity through busy shopping district, hyperrealistic 8K quality, full body portrait, paparazzi chase lighting",
   "A food critic running to restaurant reservation after getting lost, ultra-realistic 4K photography, complete portrait composition, urban restaurant district lighting, no black borders",
   "A wedding photographer sprinting to capture bride's entrance moment, photorealistic quality, 4K ultra-high definition, full figure portrait, wedding venue lighting",
   "A news reporter running to breaking story scene with microphone in hand, ultra-realistic photography, 8K resolution, complete body composition, breaking news scene lighting, no borders",
   "An archaeologist in khaki outfit running from collapsing tomb with ancient artifact, hyperrealistic 4K quality, full portrait framing, adventure archaeology lighting",
   "A spy in tuxedo running across casino floor during high-stakes chase scene, ultra-realistic photography, 4K resolution, complete figure in frame, casino floor lighting, no black borders"
]


def create_dpo_pipeline(base_model_path: str, control_lora_path: str, dpo_checkpoint: str, device: str = "cuda") -> FluxPipeline:
    """Create pipeline with base + DPO LoRA."""
    pipe = FluxPipeline.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, device=device)
    transformer = FluxTransformer2DModel.from_pretrained(
        base_model_path, subfolder="transformer", torch_dtype=torch.bfloat16, device=device
    )
    
    # Apply base EasyControl LoRA
    set_single_lora(transformer, control_lora_path, lora_weights=[1], cond_size=512)
    print("✅ Applied base EasyControl LoRA")
    pipe.load_lora_weights('XLabs-AI/flux-RealismLora')
    # NOTE: the DPO checkpoint is not applied here — variations are
    # generated with the base control model.
    pipe.transformer = transformer
    pipe.to(device)
    return pipe


def generate_variations(
    densepose_path: str,
    base_model_path: str,
    control_lora_path: str,
    dpo_checkpoint: str,
    save_folder: str,
    height: int = 1024,
    width: int = 1024,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 25,
    max_sequence_length: int = 512,
    seed: int = 42
):
    """
    Generate variations of a single densepose using different prompts.
    
    Args:
        densepose_path: Path to single densepose control image
        base_model_path: Path to base FLUX model
        control_lora_path: Path to EasyControl LoRA
        dpo_checkpoint: Path to DPO checkpoint (.ckpt file)
        save_folder: Folder to save generated images
        height: Image height (default: 1024)
        width: Image width (default: 1024)
        guidance_scale: CFG scale (default: 3.5)
        num_inference_steps: Number of steps (default: 25)
        max_sequence_length: Max sequence length (default: 512)
        seed: Random seed (default: 42)
    """
    # Create save folder
    save_dir = Path(save_folder)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load control image
    control_image = Image.open(densepose_path).convert('RGB')
    print(f"📷 Loaded control image: {densepose_path}")
    
    # Create DPO pipeline
    print("🔧 Creating DPO pipeline...")
    pipeline = create_dpo_pipeline(base_model_path, control_lora_path, dpo_checkpoint)
    
    # Generate images for each caption
    print(f"🎨 Generating {len(CAPTIONS)} variations...")
    
    for i, caption in enumerate(CAPTIONS):
        print(f"Generating {i+1}/{len(CAPTIONS)}: {caption}")
        
        # Generate image
        result = pipeline(
            caption,
            spatial_images=[control_image],
            subject_images=[],
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=torch.Generator("cpu").manual_seed(seed + i)
        ).images[0]
        
        # Save image
        output_path = save_dir / f"variation_{i+1:02d}.jpg"
        result.save(output_path)
        print(f"✅ Saved: {output_path}")
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n🎉 Generated {len(CAPTIONS)} variations in: {save_folder}")
    
    # Save caption reference
    captions_file = save_dir / "captions.txt"
    with open(captions_file, 'w') as f:
        for i, caption in enumerate(CAPTIONS):
            f.write(f"variation_{i+1:02d}.jpg: {caption}\n")
    
    print(f"📝 Caption reference saved: {captions_file}")


if __name__ == "__main__":
    fire.Fire(generate_variations) 