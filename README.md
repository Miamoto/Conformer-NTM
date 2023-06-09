# Conformer-NTM

**Note: This repository is actively maintained and being updated. Please note that the code and documentation are subject to changes and improvements. We recommend regularly checking for updates and referring to the latest version of the repository for the most up-to-date information.**

This repo contains the code and model for our paper:

Carlos Carvalho, Alberto Abad, “Memory-augmented conformer for improved end-to-end long-form ASR” in Proc. INTERSPEECH, 2023. (to appear)

## Overview

In this work, we propose a new architecture, Conformer-NTM, that combines a MANN (based on the NTM) with a conformer for E2E ASR. We demonstrated that including the external
memory is relevant to enhance the performance of the E2E ASR system for long utterances. Also, we observed that the Conformer-NTM starts to be more effective when the distribution length of the test data gets further away from the distribution length of the training data. 

The full architecture is illustrated in the figure below:

<img width="822" alt="Screenshot 2023-05-31 at 18 23 38" src="https://github.com/Miamoto/Conformer-NTM/assets/15928244/59c89af3-3ca9-4ba8-807d-4569e7196aa7">

***

The main results are summarized in this table:

<img width="790" alt="Screenshot 2023-05-31 at 18 25 02" src="https://github.com/Miamoto/Conformer-NTM/assets/15928244/b2eff7ca-ab59-4421-be9d-04adecfc6883">

## Requirements 

### Usage

**Note:** The `espnet2` directory provided in this repository includes custom modifications for training the Conformer-NTM. It contains additional memory-related functionalities and configurations specific to this work. Please ensure that you use the `espnet2` directory from this repository when setting up your environment for reproducing or extending our experiments. For any further questions, feel free to ask!  


### Pretrained models

You can download the pretrained models from the Hugging Face Model Hub:

- [Conformer-NTM-100](https://huggingface.co/Miamoto/conformer_ntm_libri_100)

Make sure to follow the instructions provided in the [repository](https://github.com/Miamoto/Conformer-NTM/blob/main/egs2/libri_ntm/asr1/README.md) to use the pretrained model in your own ASR tasks.


### Citation

Please cite our paper if you use Conformer-NTM.

```
@inproceedings{conformer-ntm,
    title={{Memory-augmented conformer for improved end-to-end long-form ASR}},
    author={Carlos Carvalho and Alberto Abad},
    booktitle={Proceedings of the 24th Annual Conference of the International Speech Communication Association (INTERSPEECH)},
    year={2023},
}
```


### Acknowledgments
