# BiaPy documentation repository
Once you have cloned this repository you will need to follow these steps in order to create the documentation site (assuming you are inside the BiaPy-doc folder).
## Step 1: Create and activate the corresponding conda environment

```
conda create -n BiaPy_doc python=3.10
conda activate BiaPy_doc
```
## Step 2: Install requirements with pip

```
pip install -r source/requirements.txt
```
## Step 3: Make the site

```
make html
```
