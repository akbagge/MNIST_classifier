# mnist_classifier

A package to use CNN to classify the MNIST dataset

## How to Run the Scripts

1. Install Requirements: Run `invoke requirements` to install all required libraries listed in `requirements.txt`. This ensures your environment is set up correctly.

2. Preprocess the Data: Run `invoke preprocess-data` to preprocess the raw MNIST data and save it in the `data/processed` folder. This will normalize the images (mean = 0, std = 1) and save the processed data as `.pt` files.

3. Train the Model: Run `invoke train` to train the CNN model on the MNIST dataset. The script will load the processed data from `data/processed` and save the trained model checkpoint in the `models/` folder.

4. Visualize the Embeddings: Run `python src/mnist_classifier/visualize.py --model-checkpoint <path_to_model_checkpoint>` to visualize the model's predictions and embeddings. Replace `<path_to_model_checkpoint>` with the path to your saved model checkpoint (e.g., `models/my_model.pth`). The script will generate a t-SNE visualization of the embeddings and save it as `embeddings.png` in the `reports/figures/` folder.



## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
