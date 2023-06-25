# Sentiment Analysis
***This repository contains code to perform sentiment analysis on text using [roBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).***

## Assumptions

- User will input a text, based on which we will predict the sentiment
- Model only preditcs 3 sentiments, Negative, Neutral, and Positive.

## Why [roBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- This version of pretrained roBERTa model preditcs 3 sentiments (Negative, Neutral, and Positive) as we need.

****
# Project Description
- Server Framework : Flask
- Deep learning Framework : Transformers
- Data Manipulation : Numpy, and Scikit-learn

# Project Setup/Installation
*Please install python for this project in your pc.*

- ### Step 1 : Create a veritual Environment
Create using any software of your choice. If you use Anaconda use the following command for creating.
```
conda create -n env_name python=3.8
```
python 3.8 is the preferred version for this project.

- ### Step 2 : Download or clone code repository
Now download the code base in local repository. You directly download download or you can use git to download the codes if you have git cli installed in your computer.
```
git clone https://github.com/ibrahim-601/sentiment_analysis_api.git
``` 
Then take your terminal to required directory.

- ### Step 3 : Dependency Installation
To install all dependency required for this project use the following command.
```
pip install -r requirements.txt
```

- ### Step 4 : Run the Project
To run the project use the following command
```
python app.py
```
This command will launch a development server in your computer at the following ip **127.0.0.1:5000**

****
## API Endpoint
- path : **127.0.0.1:5000/analyze**
- method : POST
- querystring parameter : text
    - example : text="I really like this API."

N.B: Please use this fixed format for calling this API.

- Reponse : {"sentiment": "positive/negative/neutral"}
You will receive the sentiment for the text in the response.
****

# Project Details

## Input Processing
- We need to process the data accordingly before feeding into the model.

- *analyze_sentiment()* function in the *sentiment_analyzer.py* does the processing required.

## Model Inferencing
- use *sentiment_analyzer.py* script for single inference provive the appropiate value in the *analyze_sentiment()* function.