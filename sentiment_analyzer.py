from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

# Load model at top, so that we don't need to load model for each request
MODEL_NAME = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
CONFIG = AutoConfig.from_pretrained(MODEL_NAME)
MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Preprocess text (username and link placeholders)
def preprocess(text):
    """This function removes user name and web URLs from text

    Args:
        text (str): text to remove user name and URL from

    Returns:
        str: text after removing URL and username
    """
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def analyze_sentiment(text:str)->str:
    """This function analyze sentiment of the text passed using SetFitModel's 
    sentiment analysis model found on "https://huggingface.co/StatsGary/setfit-ft-sentinent-eval"

    Args:
        text (str): text on which sentiment analysis will be performed

    Returns:
        str: Predicted sentiment of the text. "positive/negative/neutral"
    """
    try:
        # preprocess text to remove username and website links
        text = preprocess(text)
        # tokenize the processed text
        encoded_input = TOKENIZER(text, return_tensors='pt')
        # Run inference for sentiment analysis
        # we take first output as we used only one text as input
        output = MODEL(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        # sort sort to 
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        sentiment = CONFIG.id2label[ranking[0]]
    except Exception as e:
        sentiment = Exception("Model related error.")
    return sentiment

if __name__ == "__main__":
    text = "I really like the new design of your website!"
    preds = analyze_sentiment(text=text)
    print(text, " : ", preds)
    text = "His name is Bob."
    preds = analyze_sentiment(text=text)
    print(text, " : ", preds)
    text = "The new design is awful!"
    preds = analyze_sentiment(text=text)
    print(text, " : ", preds)
    text = ""
    preds = analyze_sentiment(text=text)
    print(text, " : ", preds)