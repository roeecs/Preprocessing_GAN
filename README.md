# Clustering Data for Improved GAN Results on Multi-modal Data-sets
This repo contain the following modules:
1) data preprocessing - clustering
2) iWGAN with layer conditioning for labels
3) BigGAN

This code assumes the data is in HDF5 file format.
We used the [LLD-icon-sharp](https://data.vision.ee.ethz.ch/sagea/lld/data/LLD-icon-sharp.hdf5) Logo dataset.

If the data is unlabeled, in order to run 2 & 3 modules, you will first have to preprocess your data using the clustering module. 

## Data Preprocessing - Clustering
Located in _preprocessing_clustering_ repository.
 
preprocessing for iWGAN - Run

```bash
cluster.py path_to_data h5_dataset_name output_file_path
```
preprocessing for BigGAN - Run

```bash
cluster.py path_to_data h5_dataset_name output_file_path biggan
```


## Improved WGAN & LC
Located in _iWGAN_LC_ repository.

1. Edit **_constants.py_** file to match your desired configuration.
2. Run

```bash
python train.py
```

## BigGAN
Located in _BigGAN_ repository.
1. In **_utils.py_**, add your dataset to the convenience dicts.
1. Create **_your_launch_file.sh_** configuration script in _BigGAN/scripts_ repository (e.g. **_logo_run.sh_**)
2. Run
```bash
sh your_launch_file.sh
```
