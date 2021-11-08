# DEM_ConnectedComponents
Connected Components (CC) from longitudinal river profiles for mapping of debris-flow channels.

This research and the code are associated with Mueting et al. (in press). Please cite this study when using the code.

Mueting, A., Bookhagen, B., and Strecker, M.R.: Identifying debris-flow channels using high-resolution topographic data: A case study in the Quebrada del Toro, NW Argentina

# Installation and Processing
A detailed step-by-step processing in Jupyter Notebook format is described in the example directory. In short, you will need to install an environment containing the required python packages:
```
conda create -y --name DEM_CC
conda activate DEM_CC
conda install -y jupyter ipykernel pandas numpy geopandas rasterio matplotlib scipy kneed scikit-image tqdm seaborn
```

Second, install DEM_ConnectedComponents and set path:
```
cd ~
git clone https://github.com/UP-RS-ESP/DEM_ConnectedComponents.git\n",
export PYTHONPATH=$PYTHONPATH:./~/DEM_ConnectedComponents/src
```

Third, install LSDTopoTools:
```
cd ~
wget https://raw.githubusercontent.com/LSDtopotools/LSDTT_Edinburgh_scripts/master/LSDTT_native_linux_setup.sh
sh LSDTT_native_linux_setup.sh
export PATH=$PATH:~/LSDTopoTools/LSDTopoTools2/bin
```

Now you can start using the ConnectedComponent package - see the example folder and Jupyter Notebook for a walk through on an example dataset (3 m).
