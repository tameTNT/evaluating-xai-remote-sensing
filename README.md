# Evaluating Explanation Methods for the Classification of Land Use and Land Cover

_This README was last updated 30/04/2025._

This repository contains the full source code and configuration files for the paper _"Evaluating Explanation Methods for the Classification of Land Use and Land Cover"_.

## Key Files and Results

These files are the primary files of the paper and should enable a full reproduction of the results presented.

_The program generally expects two environment variables to be set: `SAT_PROJECT_ROOT` which should ideally be set to the directory containing this `README.md` file; and `DATASET_ROOT` which is where the program will look for and download datasets to (note that some remote sensing datasets, especially multi-spectral version, can be large)._

- 🧑‍💻 `./environment_linux.yml` can be used on Linux systems to **install and recreate the exact python conda environment** used for the project via `conda env create -f environment_linux.yml`. See the [conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more information. On Windows/macOS, use `environment_fromhistory.yml` instead, although note that this may require additional debugging to be successful. Don't forget to activate the installed environment using `conda activate evaluating_xai_remote_sensing`.
- 🏋️ `./model_training/train.py` is the script used to **train all {dataset, model} combinations (referred to as DMCs) evaluated in the paper**. The script should be run from the command line from the root directory as a module: `python -m model_training.train --help` (module-like execution is required to allow for root-level local imports from within the model_training directory). Its execution can be more easily automated with arguments read from a provided file using `./run_train.sh`, e.g. `./run_train.sh model_training/config/help.args`. The exact configuration options used for each DMC are provided in `./model_training/config/`. By default, [Weights and Biases](https://wandb.ai/) is used to track training run progress (which will require a login the first time). Use the `--do_not_track` flag to disable this behaviour.
- 🕵️ `./batch_evaluate.py` is the script used to **compute all explanations and evaluation metrics**. It should be run from the command line directly using `python batch_evaluate.py --help`. Make sure to specify the path to model weights to use in `./weights_paths.json`. The exact configuration options used for each xAI method and DMC combination are provided in `./evaluate_xai/config/`. Explanations generated for batches of images are saved to `xai_output_unix` or `xai_output_windows` depending on your platform. Results are output to `./results` using the name of the `Explainer` class and provided output file name.
- 🧮 `./results/all_results_export.xlsx` contains the **individual class label-level results** summarised and discussed in the paper, collated from the 3 (one for each xAI method) raw `.h5` files in the same directory each containing 12 `pandas` `DataFrame` objects - one for each DMC.
- 📊 `./notebooks/raw_results_exploration.ipynb` and `./notebooks/visualisations_for_report` contain all the graphics (and others besides) used in the report alongside more detailed data exploration by model type/dataset/etc. and further observations where appropriate.

## Key Directories

This section documents the key libraries developed for the use primarily in the aforementioned scripts. For those seeking to investigate or build on the precise implementations of the methods described in the paper, this is a helpful guide to where to find them. Further documentation on each class, any functions, and their use is provided in the source code itself.

- `./dataset_processing/` – Contains wrapper classes around `torchgeo` dataset objects via the `core.RSDatasetMixin` class, implementing data transformation, normalisation, and augmentations alongside adding attributes expected throughout the rest of the program. Use `get_dataset_object(...)` to get a dataset object containing images.
- `./evaluate_xai/` – Contains the implementations of each evaluation metric, grouped by the Co-12 property they fall under. All properties subclass `Co12Property` and have an `.evaluate(...)` method with a `method` argument (alongside `**kwargs`) to specify which metric to calculate (and how). The directory also includes `deletion.py` providing the implementation of the different deletion/pixel perturbation methods detailed in the paper and used across several metrics.
- `./helpers/` – Includes various helper and utility modules including most importantly `log.py` for logging throughout all files and `plotting.py` for showing images and explanation heatmaps in a grid flexibly, including multispectral images.
- `./models/` – Implements wrappers around `torchvision` models via the `core.Model` class with methods to load weights (`load_weights`) freeze particular layers (`freeze_layers`) and get the target layers for CAM-based explanations (`get_explanation_target_layers`). Each `Model` subclass (e.g. `convnext.ConvNeXtSmall`) implements loading ImageNet pretrained model weights and adjusting the input and output layers of the model to match the dataset.
- `./xai/` – Provides wrappers around the implementations of PartitionSHAP, Grad-CAM, and KPCA-CAM provided by the `shap` and `pytorch-grad-cam` packages via the `Explainer` class. This class implements saving and loading explanations to disk, creating ranked explanations automatically, and combining `Explainer` objects together via the `|` or operator among other things.


### Additional Notes

- ⚠️ The notebooks in `./notebooks/old_experiments/` have not been maintained through API changes in the rest of the project so these may not function as expected or even run at all.
- The demo data (`./results/GradCAM/demo_output*`, `./notebooks/demo_results.ipynb`) was generated on an M2 MacBook Pro for {EuroSATRGB, ResNet50} on GradCAM via `batch_evaluate.py`.
- The final archive zip export was created using `7zz a ../evaluating_xai_remote_sensing__source_code.zip batch_evaluate.py dataset_processing environment_* evaluate_xai helpers LICENSE model_training models notebooks README.md results run_train.sh standalone_evaluation.py xai .gitignore '-xr!*.DS_store'`.
