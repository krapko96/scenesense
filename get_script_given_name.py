import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import IMSDbLoader

def scrape_script_given_name(movie_name):
    """
    Scrapes the list of movie titles from the IMSDB all scripts page.

    Returns:
        list: A list of strings, where each string is a movie title.
              Returns an empty list if scraping fails.
    """
    url = f'https://imsdb.com/scripts/{movie_name}.html'

    loader = IMSDbLoader(url)
    data = loader.load()
    return data



if __name__ == "__main__":
    script = scrape_script_given_name("10-Things-I-Hate-About-You")
    page_content = script[0].page_content
    cleaned_script_split_join = " ".join(page_content.split())
    with open('script.txt', 'w') as f:
        f.write(cleaned_script_split_join)
