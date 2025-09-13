# 3D Animals Codebase

This repository contains the unified codebase for cheetah pose estimation on articulated 3D animal reconstruction and motion generation, including:

- [Learning the 3D Fauna of the Web](https://kyleleey.github.io/3DFauna/) (CVPR 2024) [![arXiv](https://img.shields.io/badge/arXiv-2401.02400-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2401.02400) - a pan-category single-image 3D quadruped reconstruction model


## Installation
See [INSTALL.md](./INSTALL.md).


## Run
Once the data is prepared, both training and inference of all models can be executed using a single command:
```shell
python run.py --config-name CONFIG_NAME
```
or for training with DDP using multiple GPUs:
```shell
accelerate launch --multi_gpu run.py --config-name CONFIG_NAME
```
`CONFIG_NAME` can be any of the configs specified in `config/`, e.g., `test_fauna` or `train_fauna`.

### Testing using the Pretrained Models
The simplest use case is to test the pretrained models on test images. To do this, use the configs in `configs/` that start with `test_*`. Open the config files to check the details, including the path of the test images.

Note that only the RGB images are required during testing. The DINO features are not required. The mask images are only required if you wish to finetune the texture with higher precision for visualization (see [below](#test-time-texture-finetuning)).

When running the command with the default test configs, it will automatically save some basic visualizations, including the reconstructed views and 3D meshes. For more advanced and customized visualizations, use `visualization/visualize_results.py` as explained [below](#visualization).

### Visualization
We provide some scripts that we used to generate the visualizations on our project pages. To render such visualizations, simply run the following command with the proper test config:
```shell
python visualization/visualize_results_fauna_clap.py --config-name test_fauna
```

Check the `#Visualization` section in the config files for specific visualization configurations.

#### Rendering Modes
The visualization script supports the following `render_modes`, which can be specified in the config:
- `input_view`: image rendered from the input viewpoint of the reconstructed textured mesh, shading map, gray shape visualization.
- `other_views`: image rendered from 12 viewpoints rotating around the vertical axis of the reconstructed textured mesh, gray shape visualization.
- `rotation`: video rendered from continuously rotating viewpoints around the vertical axis of the reconstructed textured mesh, gray shape visualization.
- `animation` (only supported for quadrupeds): two videos rendered from both a side viewpoint and continuously rotating viewpoints of the reconstructed textured mesh animated by interpolating a sequence of pre-configured articulation parameters. `arti_param_dir` can be set to `./visualization/animation_params` which contains a sequence of pre-computed keyframe articulation parameters.
- `canonicalization` (only supported for quadrupeds): video of the reconstructed textured mesh morphing from the input pose to a pre-configured canonical pose.

#### Test-time Texture Finetuning
To enable texture finetuning at test time, set `finetune_texture: true` in the config, and (optionally) adjust the number of finetune iterations `finetune_iters` and learning rate `finetune_lr`.

For more precise texture optimization, provide instance masks in the same folder as `*_mask.png`. Otherwise, the background pixels might be pasted onto the object if shape predictions are not perfectly aligned.


## 3D-Fauna [![arXiv](https://img.shields.io/badge/arXiv-2401.02400-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2401.02400)
[3D-Fauna](https://kyleleey.github.io/3DFauna/) learns a pan-category model for single-image articulated 3D reconstruction of any quadruped species.

### Training
To train 3D-Fauna on the Fauna Dataset, simply run:
```shell
python run.py --config-name train_fauna
```


## Citation
If you use this repository or find the papers useful for your research, please consider citing the following publications, as well as the original publications of the datasets used:

```
@InProceedings{li2024fauna,
  title     = {Learning the 3D Fauna of the Web},
  author    = {Li, Zizhang and Litvak, Dor and Li, Ruining and Zhang, Yunzhi and Jakab, Tomas and Rupprecht, Christian and Wu, Shangzhe and Vedaldi, Andrea and Wu, Jiajun},
  booktitle = {CVPR},
  year      = {2024}
}
```

## modify files
```
./run.py
./config/test_fauna.yaml
./config/test_fauna_clap.yaml
./model/dataset/SequenceDataset.py
./model/dataset/util.py
./model/models/AnimalModel.py
./model/models/FaunaWithAudio.py
./visualization/extract_touch_point.py
./visualization/visualize_results_fauna_clap.py
./INSTALL.md
```
## An overview of changes and implementations for adding audio loss to the Fauna model:

  1. Created a new model file: FaunaWithAudio.py, which inherits from the original AnimalModel and extends the FaunaModel.
  2. Added audio loss configuration: Defined the AudioLossConfig data class to manage configuration parameters for audio-related losses.
  3. Integrated the CLAP model: Initialised the CLAP model and processor within the new model for audio feature extraction.
  4. Implemented audio feature extraction method: The `extract_audio_features` method utilises the CLAP model to extract features from audio files.
  5. Implemented two audio loss functions:
    - `compute_audio_consistency_loss`: Calculates audio-visual consistency loss, encouraging predicted 3D parameters to maintain semantic coherence with the audio.
    - compute_audio_sync_loss: Calculates the audio-visual synchronisation loss, encouraging predicted temporal parameters (such as joint parameters) to align temporally with the audio.
  6. Integrated audio loss computation into forward propagation: Modified the forward method to compute audio losses at appropriate points and add them to the total loss.

To validate this implementation and test the effects of audio loss, we need to carry out the following steps:

  1. Install CLAP: Ensure the CLAP library is installed in the project environment.
  2. Modify the dataset: Update the dataset loader to enable loading audio file paths corresponding to video sequences.
  3. Update configuration files: Create or modify model configuration files, adding settings related to audio loss.
  4. Train the model: Commence the training process using the new model and configuration, observing the impact of audio loss on training outcomes.
