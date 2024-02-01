# Pull Request Analysis

## Installation
Make sure you have Python 3 installed. 

To install the requirements, run `pip install -r requirements.txt`

## Getting the Code Data
To get the code for each pull request, run `python scripts/get_code.py` which will get the code associated with each pull-request and save it into a new csv file called `pullreq_with_code.csv`

When running this script, be sure to replace the `github_token` variable with your GitHub token. A GitHub token can be generated here: https://github.com/settings/tokens

## Training the Model on the Data
All scripts are in `scripts/` folder. To train the model using the Random Forest model, you can run `jupyter notebook`in the command line and open up the `train_random_forest.ipynb` notebook and run the file.

To train the BERT model, you can run `jupyter notebook`, open up the `train_bert.ipynb` notebook and run the file.