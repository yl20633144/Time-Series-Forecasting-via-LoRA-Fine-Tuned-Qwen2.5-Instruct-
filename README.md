## Description
This project investigates the use of large language models for time series forecasting. We apply LoRA to fine-tune Qwen2.5-Instruct (0.5B) on the Lotka–Volterra predator-prey system to evaluate parameter-efficient adaptation for scientific forecasting tasks.

## Conda environment

There is a conda environment file that will set up the required packages and libraries for this coursework. You can create a conda environment from this file with:

```bash 
conda env create -n <name> --file environment_m2.yml
```

### Activate the environment:

```bash 
conda activate <name>
```

## License
This project is licensed under the MIT License.


## Structure
### Python scripts:
All python scripts are placed inside /src:
1. lora_skeleton.py: It includes functions for loading and chunking datasets, LoRA linear class, applying LoRA adaption to a model and training a model
2. preprocesser.py: It includes functions to preprocess raw data into string elements. In addition, the string dataset is split into three train,validation and test sets. 
3. qwen.py: load the original model and tokenizer.
4. flops.py: It includes a set of functions to calculate the flops consumption. 
5. Evaluation.py: It includes functions for context and target tokens splitting, predicted tokens decoding and the procedure of model evaluation.
The documentation is placed in /docs/build/html.


### Python notebooks:
Qwen_base_codes.ipynb is a python notebook for:
1. The evaluation process of the untrained model.
2. Using flops.py, calculate the flops consumption per experiments.
3. The training process of 3a, which is the default-parameter trial training with 500 steps.
4. Evaluation of the trained model.

Qwen_grid_search_colab.ipynb is a python notebook for:
1. The grid search part of 3b, including 9 training and 9 evaluations
2. Three additional training with different context length (with evaluations).
3. The final training with longer steps (with evaluations).

### Report:
/report contains the final report for this cooursework called "main".

### Trajectories:
This a folder containing some plots of trajectories from grid search to the final longer experiment, in order to provide more evidences for the prediction ability.

## Notes:
1. Random seeds were set for reproducibility.
2. Qwen_grid_search_colab.ipynb is implemented using Colab. If a re-run will be required, please change the path to your own Google drive.
3. In addition, wandb visualization is implemented inside functions. To re-run the codes, please login to your own wandb account.
4. Weights and biases for experiments in 3b and 3c are saved in the folder "Weights". For future new evaluation, no re-train will be required as the weights can be loaded.


## AI generation tools:
Declaration of AI generation tools, ChatGPT, for the coding part is made here:
1. It helps me format my code comments and docstrings, making the codes reader-friendly.
2. It helps me implement the FLOPs calculation as well as the model architechture.
3. It helps me make plots with good quality.
4. It helps me debug problems in my codes.
5. It helps me format the README.md, pyproject.toml and autodocumentation.