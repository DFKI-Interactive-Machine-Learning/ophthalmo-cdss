# License
<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/DFKI-Interactive-Machine-Learning/ophthalmo-cdss">Ophthalmo-CDSS</a> by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://github.com/robertleist">Robert Andreas Leist</a> is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt=""></a></p>

# Backend Documentation
## Overview
This document provides an overview of the backend of the OphthalmoDashboard. The backend is responsible for the communication between the frontend and the data. It is implemented in the `src/` folder of the repository.
The backend consists of several parts:
- `data/`: Functionality concerning data base access, preprocessing and loading.
- `prognosis/`: Time series forecast model architectures, weights and training scripts.
- `segmentation/`: YNet architecture and weights. This folder is a mirror of the [YNet repository](https://github.com/azadef/ynet).
- `alignment.py`: Contains the code for aligning the OCT images.
- `recommendation.py`: Contains the code for the recommendation system.
- `reconstruction.py`: Contains the code for the reconstruction of OCTs.
- `util.py`: Contains utility functions used by the backend.
- `visualisation.py`: Contains functions for visualizing the data.
- `vol.py`: Contains a VOL class for loading OCTs in the VOL format. This class was adapted from the [Heidelberg Engineering Heyex VOL reader](https://github.com/ayl/heyexReader). It features an automatic segmentation of the OCT and computes three-dimensional reconstructions from segmentations.

## Prognosis
The prognosis module contains the time series forecast model architectures, weights and training scripts. 
The module contains the following files:
- `architectures.py`: Contains the model architectures for the time series forecast. Implemented architectures are:
  - BasicRNNRegressor: A simple RNN model.
  - BasicLSTMRegressor: A simple LSTM model. Consisting of configurable LSTM layers and a fully connected layer.
  - AdvancedLSTMRegressor: Slightly advanced LSTM model. Consisting of configurable LSTM layers, a layer normalization layer and a fully connected layer.
- `models.py`: Contains convenience classes to combine models of varying time targets. Classes also automatically load weights from the `prognosis/weights/` folder.
- `train.py`: Contains the training script for the time series forecast models.
- `weights/`: Contains the weights for the time series forecast models.
  - Contains a subfolder for each predicted metric `n_fluids`, `v_fluids` and `visual_acuity`.
    - Each folder named after the predicted metric contains subfolders, which contain weights for different time targets of the forecast. The naming convention for the subfolders is "`interpolation_method`\_`metric`\_`time_target`".
      - The weights are stored as `.pt` files. The naming convention for the weights is "`model_name`\_`parameters`.pt". For example: `AdvancedLSTMRegressor_hidden_16_layers_2_lr_0.01_redf_0.9_batch_128_loss_L1Loss_drop_0.1.pt` is an AdvancedLSTMRegressor model with 16 hidden layers, 2 LSTM layers, a learning rate of 0.01, a reduction factor of 0.9, a batch size of 128, a loss function of L1Loss and a dropout rate of 0.1.
    - Each folder also contains the results of a training run for the different time targets. The naming convention here is "train_results_`time_target`.csv". The tables contain the training and validation loss and hyperparameters for different models and are created by the `train.py` script.
  - Computer StandardScalers are also saved in this folder. They are used to scale the input data for the models and inverse scale the output. Their naming convention is "`interpolation_method`\_`metric`\_`time_target`\_`x or y depending on whether features or target should be scaled`_scaler.pkl".
## Segmentation
The segmentation module contains the YNet architecture and weights. This folder is a mirror of the [YNet repository](https://github.com/azadef/ynet).
Only a folder with the weights is included in this repository. 
The weights are stored in the `segmentation/pretrained_models/` folder. The weights are stored as `.pt` files.
Convenience classes for loading the weights are provided in the `segmentation/__init__.py` file.
