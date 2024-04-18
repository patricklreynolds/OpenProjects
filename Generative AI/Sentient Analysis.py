from textblob import TextBlob
from newspaper import Article

url = 'https://en.wikipedia.org/wiki/Artificial_intelligence'
article = Article(url)

article.download()
article.parse()
article.nlp()

text = article.summary
print(text)

blob = TextBlob(text)
sentiment = blob.sentiment.polarity # -1 to 1
print(sentiment)
