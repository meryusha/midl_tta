1. **Data**
    We provide the sharded videos and pre-trained checkpoints for you. 

    Please find the link with to dowloaded the sharded files. Please, unzip the folders and place them in this working repo. 


    You should have the following folders inside of your working repo:
    - EPIC_shards
    - EPIC_sounds_shards
    - checkpoints

    NOTE: if you don't want to download the data for now, you can still test the code by only downloading **checkpoints** folder. It contains all prediction files so you can just run the evaluation code on them. 

    Below we clarify on how we obtained the videos in case you want to repeat this process: 

    a. Downloading the videos and annotations
    
     - You will need to dowload EPIC-KITCHENS and EPIC-SOUNDS datasets

    - For doing that, we recommend using the official epic-kitchens download script from here https://github.com/epic-kitchens/epic-kitchens-download-scripts

    - You can find the annotations for Epic-Kitchens here 

    - And the annotations for Epic-Sounds here https://github.com/epic-kitchens/epic-sounds-annotations

    b. Video pre-processing: trimming and sharding

    - Keep in mind that Epic-Kitchens and Epic-Sounds videos are long untrimmed videos. We trim those videos into short clips, according to the annotated segments. You can refer to the following scripts to trim and shard the videos. To trim use : 

        - development_scripts/trimming/epic/trim_epic_sound.py # to trim Epic-Sounds videos
        - development_scripts/trimming/epic/trim_epic.py  # to trim Epic-Kitchens videos


2. **Environment**
    Please install the env from the yaml file as
    conda env create -f environment.yml
    
    Activate your environemnts once it has been installed
    Should be something like:
        conda activate midl_tta

3. **Running the TTA inference with the baseline TTA methods: TENT, SHOT, ETA:**

    To run the inference for EPIC-SOUDNS, use the following command:

        bash scripts/inference/tta_sounds/tta_inference_epic_sound_baselines.sh

    To run the inference for EPIC-SOUDNS, use the following command:

        bash scripts/inference/tta_epic/tta_inference_epic_tta_baselines.sh

    You can modify, which TTA method you want to  use by updating the METHOD variable.

    You can modify, which missing ratio you want to evaluate with by updatin the PROP variable.

    You don't need to update the SEED varibale. 

4. **Running th TTA inference with MiDL:**

    To run the inference for EPIC-SOUDNS, use the following command:

        bash scripts/inference/tta_sounds/tta_inference_epic_sound.sh

    To run the inference for EPIC-SOUDNS, use the following command:

        bash scripts/inference/tta_epic/tta_inference_epic.sh

    You can modify, which TTA method you want to  use by updating the METHOD variable.
    You can modify, which missing ratio you want to evaluate with by updatin the PROP variable.
    You don't need to update the SEED varibale. 
    Currently, we included the prediction files, so you don't actually need to run the inference to see the results. However, if you want to re-run the TTA process, please delete or rename the method folder, e.g. if you want to run the MiDL on EPIC-KITCHENS, delete/rename the checkpoints/EPIC-KITCHENS/midl folder 
    Then, run

        bash scripts/inference/tta_sounds/tta_inference_epic_sound.sh