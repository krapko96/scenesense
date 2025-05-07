import requests
from bs4 import BeautifulSoup

def scrape_imsdb_titles():
    """
    Scrapes the list of movie titles from the IMSDB all scripts page.

    Returns:
        list: A list of strings, where each string is a movie title.
              Returns an empty list if scraping fails.
    """
    url = "https://imsdb.com/all-scripts.html"
    movie_titles = []
    directors = []
    movies_and_directors = []

    try:
        # Fetch the content of the page
        response = requests.get(url)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # The movie titles are within <p> tags, each containing an <a> tag
        # We find all <p> tags and then extract the text from the <a> tag within them
        # We can be more specific by looking for <p> tags that contain an <a> tag
        # or by observing the structure, the titles are directly under a specific <td>
        # Let's refine based on typical IMSDB structure - they are usually in <p> tags
        # directly under the main content area. Looking at the source, they are in
        # <td valign="top"> followed by a series of <p><a href="...">Title</a></p>

        # Find the table data cell containing the list of scripts
        # Based on inspection, the main list is within a <td> with valign="top"
        # and width="97%" inside the main table.
        td_with_scripts = soup.find_all('p')
        for movie in td_with_scripts:
            # Each movie title is within an <a> tag inside a <p> tag
            a_tag = movie.find('a')
            i_tag = movie.find('i')
            if a_tag and a_tag.text:
                title = a_tag.text.strip() if i_tag else "Unknown"
                director = i_tag.text.strip() if i_tag else "Unknown"

                movie_titles.append(title)
                directors.append(director)
                movies_and_directors = list(zip(movie_titles, directors))

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return []
    except Exception as e:
        print(f"An error occurred during parsing: {e}")
        return []

    return movie_titles, directors, movies_and_directors

if __name__ == "__main__":
    titles, directors, movies_and_directors = scrape_imsdb_titles()
    if titles:
        print(f"Successfully scraped {len(titles)} movie titles:")
        # Print the first 20 titles as an example
        for i, title in enumerate(titles[:20]):
            print(f"{i+1}. {title}")
        if len(titles) > 20:
            print("...")
    else:
        print("Could not retrieve movie titles.")

    with open('movie_titles_and_directors.txt', 'w') as f:
        for item in movies_and_directors:
            f.write(item[0] + " | " + item[1] + "\n")
