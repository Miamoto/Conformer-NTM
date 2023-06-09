
# External Memory -> Conformer-NTM

This repository serves as a starting point for using the Conformer-NTM model with ESPnet. The experiments described here utilize the LibriSpeech corpus as an example, but you can use a different corpus if desired.

To perform the experiments, follow the steps below:

1. **Create your own ESPnet recipe**: Customize the recipe by setting the desired task and directory name. Execute the following commands:
    ```bash
    % task=asr1
    % name=      # use the name you want for your directory
    % egs2/TEMPLATE/asr1/setup.sh egs2/${name}/asr1
    ```
   For more details, refer to the [ESPnet Template](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/README.md) documentation.

2. **Copy necessary files**: Copy the following files and directories provided in this repository to `egs2/${name}/asr1`:
   - `run.sh`
   - `conf` directory
   - `local` directory

3. **Create the LibriSpeech data directory**: Run the following command:
    ```bash
    % ./run.sh --stage 1 --stop-stage 1 &
    ```

4. **Data preparation**: Execute steps 2, 3, 4, and 5 for data preparation.

5. **Model Training and Inference**:

   a) If you want to skip training and perform inference directly you can use the provided pretrained [Conformer-NTM-100](https://huggingface.co/Miamoto/conformer_ntm_libri_100) model available on Hugging Face:
   
    ```bash
    ./run.sh --skip_data_prep true --skip_train true --download_model miamoto/conformer_ntm_libri_100
    ```
   
   b) If you want to train the model from scratch, perform step 10 and then step 11. Ensure that you have added the `use_memory` variable to the `asr.sh` file in your recipe to indicate whether memory will be used during training. Check the provided `asr.sh` file to see where it is used:
   
    ```bash
    ${python} -m espnet2.bin.asr_train \
        --use_preprocessor true \
        ...
        --use_memory ${use_memory} \ 
        --output_dir "${asr_exp}" \
        ${_opts} ${asr_args} 
    ```

   Additionally, if you want to train with external memory, include the memory parameters in the config file. For example:

   ```yaml
   ntm: ntm
   ntm_conf:
       memory_num_rows: 256
       memory_num_cols: 10
       num_heads: 1
       max_shift: 1
   ```

   Remove these parameters from the configuration file if you are not using external memory. Check the provided configuration files under (`Conformer-NTM/egs2/libri_ntm/asr/conf/tuning/`) directory for reference.

6. After training your model, perform inference by running steps 12 (decoding) and 13 (scoring). If your trained model does not have external memory, comment out the following line (`self.asr_model.initialize_ntm(self.device, batch_size)`) in the provided `espnet2/bin/asr_inference.py` file. Uncomment the line if your model has external memory.


If you have any further questions, feel free to ask!
