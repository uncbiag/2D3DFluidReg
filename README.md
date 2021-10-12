# lung_sdt_ct

This is the repository for paper [Fluid Registration Between Lung CT and Stationary Chest Tomosynthesis Images](https://drive.google.com/file/d/1-gORB0x9qa8hDpnpLSISXGmb9I6j9SG9/edit), MICCAI 2020

![Model Structure](https://github.com/uncbiag/lung_sdt_ct/readme_materials/miccai_fig1.png)


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
To run the code, two setting files are required. One is for configuration of the mermaid library. The other is configuration file of the data. Sample setting files can be found at [here](https://github.com/uncbiag/lung_sdt_ct/settings/dirlab)
```
cd lung_sdt_ct
python -s ./settings/dirlab/lung_registration_setting_dct1.json -d ./exp/test -p ./exp/test/preprocessed
```