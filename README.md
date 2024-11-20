# README

This is the code repository for master thesis project "Improving atherosclerotic plaque segmentation and estimating their clinically relevant parameters" at the Faculty of Electrical engineering at CTU in Prague.

## Implementation details, requirements & installation
The work is primarily implemented in Python 3 and PyTorch. The recommended version of Python is >=3.7. To a standard conda environment, you will need PyTorch (version >=1.6 id needed to load the models properly) and Torchvision (version >=0.9). 

For the server using ml loading use these commands to install dependencies:

ml Python/3.9.6-GCCcore-11.2.0 \
ml PyTorch/2.0.0-foss-2022a-CUDA-11.7.0 \
ml torchvision/0.15.1-foss-2022a-CUDA-11.7.0 \
ml scikit-learn/1.1.2-foss-2022a \
ml OpenCV/4.7.0-foss-2022a-CUDA-11.7.0-contrib  
ml scikit-image/0.19.3-foss-2022a
ml matplotlib
ml plotly.py

## Usage 

### Training segmentaion
To run the segmentation the images and true-segmentations  have to be in train/val/test directories now it is set that it works when data are in: segmentation/data/train_val_test/["train" or "test" or "val"]/["images" or "segmentations"]

To run the segmentation training, fill the parameters in the segmentation/train_segmentation.py and then use command:

```python segmentation/train_segmentation.py --device [free cuda device]```

fill the [free cuda device] with available device (0 to 7 on cmp servers), the resulting model and results are available at the folder "models/[network type]/[experiment name]"

### using segmentation network for key images dataset
to create segmentation - images pair do following
- put raw images to data/carotid_key_inpainted_dataset/images
- put forecast_key_dataset.csv into data/
- in the create_localization_segmentation.py spcify segmenation algorithm parameters
- run: python create_localization_segmentation.py --device [number of the cuda device]
- results are in data/carotid_key_inpainted_dataset/[experiment name]

### creating synthetic dataset 
to create synthetic dataset specify parameters in image_generation/run_data_generation.py and run:
``` python image_generation/run_data_generation.py ```

### Training classification or reggresion networks
To run the calssification the images and segmentations  have to be in train/val/test directories now it is set that it works when data are in: parameters_computation/data/[experiment name from from running create_localization_segmentation.py]/["train" or "test" or "val"]/["images" or "segmentations"]
also the csv with annotations have to be availabale: in my example as parameters_computation/data/forecast_key_dataset.csv

To run the parameters computation training, fill the parameters in the parameters_computation/train_nongeom.py for classification of non-geometric paramteres or parameters_computation/train_regression.py for regressing plaque width or parameters_computation/synthetic_train_nongeom.py for synthetic dataset echogenecity and then use command:

```python parameters_computation/train_nongeom.py --device [free cuda device]``` 
```python parameters_computation/train_regression.py --device [free cuda device]```
```python parameters_computation/synthetic_train_nongeom.py --device [free cuda device]```

fill the [free cuda device] with available device (0 to 7 on cmp servers), the resulting model and results are available at the folder "models/[experiment name]"

### direct computation od plaque width 
the notebook that was used to compute the width and show results is in: notebooks/evaluate_plaque_width_from_segmentation.ipynb


