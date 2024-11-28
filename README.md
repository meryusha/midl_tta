1. DATA
    We provide the sharded videos for you. 
    Please find the link with to dowloaded the sharded files. Please, unzip the folders and place them in this working repo. 
    You should have the following folders:
    - EPIC_shards
    - EPIC_sounds_shards
    Just to clarify on how we obtained the videos in case you want to repeat this process.
    a. Downloading the videos and annotations
        You will need to dowload EPIC-KITCHENS and EPIC-SOUNDS datasets
        For doing that, we recommend using the official epic-kitchens download script from here https://github.com/epic-kitchens/epic-kitchens-download-scripts
        You can find the annotations for Epic-Kitchens here 
        And the annotations for Epic-Sounds here https://github.com/epic-kitchens/epic-sounds-annotations
    b. Video pre-processing: trimming and sharding
        Keep in mind that Epic-Kitchens and Epic-Sounds videos are long untrimmed videos. We trim those videos into short clips, according to the annotated segments. You can refer to the following scripts to trim and shard the videos. To trim: 
        - development_scripts/trimming/epic/trim_epic_sound.py # to trim Epic-Sounds videos
        - development_scripts/trimming/epic/trim_epic.py  # to trim Epic-Kitchens videos
    c. To download the pre-trained baseline checkpoint: 
                        

2. Environment
    Please install the env from the yaml file as
    conda env create -f environment.yml

3. Running the TTA inference with the baseline TTA methods: TENT, SHOT, ETA:

    To run the inference for EPIC-SOUDNS:

4. Running th TTA inference with MiDL:

