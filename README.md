# 2D/3D Fluid Registration

This is the repository for paper "Fluid Registration Between Lung CT and Stationary Chest Tomosynthesis Images" MICCAI 2020
[Paper](https://drive.google.com/file/d/1-gORB0x9qa8hDpnpLSISXGmb9I6j9SG9/edit),[Presentation](https://youtu.be/rPVmPg1rXSI)

In this work we propose a differentiable projection operator which renders 2D projections from a 3D volume with given geometry parameters (pose of the emitters).
![Differentiable projectionoperator](/readme_materials/miccai_fig2.png)
This operator can be used in a 2D3D deformable registration framework as shown below. 
![Model Structure](/readme_materials/miccai_fig1.png)



# Installation
1. Setup environment
    ```
    cd lung_sdt_ct
    conda env create -f environment.yml
    ```
2. Install [Mermaid](https://github.com/uncbiag/mermaid)
3. Install this repo
    ```
    python setup.py develop
    ```

# How to run
To run the code, two setting files are required. One is for configuration of the mermaid library. The other is configuration file of the data. Sample setting files can be found at [here](https://github.com/uncbiag/lung_sdt_ct/tree/dev/settings/dirlab)
```
cd lung_sdt_ct
python registration/test_ct_registration.py -s ./settings/dirlab/lung_registration_setting_dct1.json -d [OUTPUT_FOLDER] -p [PREPROCESSED_DATA_OUTPUT_FOLDER]
```
