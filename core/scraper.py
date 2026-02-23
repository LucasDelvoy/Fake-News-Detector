from newspaper import Article

def scraper(url):
    article = Article(url)
    article.download()
    title = article.title
    text = article.text
    return title, text