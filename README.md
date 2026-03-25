# BiaPy documentation repository
Once you have cloned this repository you will need to follow these steps in order to create the documentation site (assuming you are inside the BiaPy-doc folder).
## Step 1: Create and activate the corresponding conda environment

```
conda create -n BiaPy_doc python=3.11
conda activate BiaPy_doc
```

## Step 2: Install requirements with pip

```bash
pip install -r source/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Step 3: Make the site

```
make html
```

After executing these steps, the site should appear under `./build/html/`.
