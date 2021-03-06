# Sentiment analysis of tweets

This is the code for my Master's thesis on sentiment analysis.

## Using the code

The code uses ML libraries like Scikit-learn, XGBOOST and NLP libraries like gensim and nltk. A requirements.txt is provided to make requirements installation easier. Please check virtualenv for creation of python environments for this project. If you use python for DS, probably most of dependencies are already met.

Create a virtual environment and activate it:

    virtualenv pythonenv
    source pythonenv/bin/activate

I have put the virtual environment into another folder, so you could use the get_env.sh to activate on this new directory if needed:

     source get_env.sh

Run the requirements.txt:

    pip install -r requirements.txt

Download the resources needed:

    python download_resources.py

PS: the resources may or not be freely available on the net. Please check the authors regulation and cite their work.

## Folders

Description of the folders

    0-Scripts: Scripts for preprocessing, analysing and predicting
    1-Input: Input files as originals
    2-Processed: Files generated by preprocessing scripts
    3-Output: Files generated by the predicting scripts
    4-Resources: Additional resources used, like dictionaries and jars for processing the files
    5-Setup: scripts to download resources
    6-Models: models may be saved here for later use
    7-SemEval: Sem eval files
    

The folders may contain additional readme.md files explaining their contents.

## Additional links

### Python stuff

- installing scikit on pythonenv: https://gist.github.com/fyears/7601881