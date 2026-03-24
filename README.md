# BiaPy documentation repository
Once you have cloned this repository you will need to follow these steps in order to create the documentation site (assuming you are inside the BiaPy-doc folder).
## Step 1: Create and activate the corresponding conda environment

```
conda create -n BiaPy_doc python=3.11
conda activate BiaPy_doc
```
## Step 2: Install requirements with pip

For Linux/Windows:

```
pip install -r source/requirements.txt
```

For macOS (remove CUDA-only packages before installing):

```
grep -Ev '^(nvidia-.*-cu11|triton)' source/requirements.txt > source/requirements.macos.txt
pip install -r source/requirements.macos.txt
```

> **Troubleshooting (autodoc imports)**
> If `make html` fails with `ModuleNotFoundError` for `torchinfo` or `torchmetrics`,
> install them in the same environment and run the build again:
>
> ```bash
> pip install torchinfo torchmetrics
> make html
> ```
## Step 3: Make the site

```
make html
```

After executing these steps, the site should appear under `./build/html/`.
