The datasets referenced by the user are extremely large (over 2 GB in total). They cannot be stored directly in this repository. To download them locally, run:

```bash
pip install gdown

# All datasets are now hosted in a single Google Drive folder.
gdown 'https://drive.google.com/drive/folders/1iZE_Cg5wAk_94Uk1DgNrOLiqp4F6cbfZ?usp=sharing' --folder
```

This will recreate the `data` and `dataset` directories with the same structure as provided in the Google Drive folders.
