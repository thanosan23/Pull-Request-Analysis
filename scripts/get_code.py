# import libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests
import concurrent.futures

# importing polars to read in the dataset quickly
import polars as pl

# using polars to read in the dataset
data = pl.read_csv("../data/new_pullreq.csv").to_pandas()

# function that can read all the added code from a pull request and return it as a block of code
def get_added_lines(diff):
    return '\n'.join(line[1:] for line in diff.split('\n') if line.startswith('+'))

# uses the open source github api to read in the pull request code
def get_pull_request_files(owner, repo, pull_number, github_token):
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/files"
    headers = {
        'Accept': 'application/vnd.github+json',
        'Authorization': f'Bearer {github_token}',
        'X-GitHub-Api-Version': '2022-11-28',
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

# get the added code from the pull request
def get_added_code(owner, repo, pull_number, github_token):
    files = get_pull_request_files(owner, repo, pull_number, github_token)
    for file in files:
        return get_added_lines(file['patch'])

# enter your github token here
github_token = 'ENTER GITHUB TOKEN HERE'

# create a list of invalid repos (repositories that are no longer available, are private, etc.)
invalid_repos = []

# function that gets the code from the pull request and adds it to the dataframe
def get_code(row_id):
    # get information about pull request
    owner = data.iloc[row_id]["ownername"]
    reponame = data.iloc[row_id]["reponame"]
    github_id = data.iloc[row_id]["github_id"]

    # try getting the code from the pull request, if not (error with request due to invalid pr), add to invalid repos
    try: 
        code = get_added_code(owner, reponame, github_id, github_token)
        data.loc[row_id, "added_code"] = code
    except:
        invalid_repos.append(github_id)

# iterate through all the pull requests and get the code
i = 0
max_val = data.shape[0]
# run the code in parallel
with concurrent.futures.ThreadPoolExecutor() as e:
    fut = [e.submit(get_code, i) for i in range(0,  max_val)]
    for r in concurrent.futures.as_completed(fut):
            print(f"Finished {i}")
            i+=1

# create a new dataframe with the added code
data.to_csv("../data/pullreq_with_code.csv")