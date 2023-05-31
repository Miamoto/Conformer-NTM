# Conformer-NTM

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

### Pretrained models

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



