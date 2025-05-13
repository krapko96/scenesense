import os
from flask import Flask, render_template, request, redirect, url_for, jsonify

# Langchain specific imports
# Use correct imports based on your Langchain/community package version
try:
    # Newer versions might be in langchain_community
    from langchain_community.document_loaders import IMSDbLoader
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains.question_answering import load_qa_chain
    from langchain.chains import LLMChain
    from langchain.schema import Document
    from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
except ImportError:
    # Fallback for older versions if needed, adjust based on your install
    print("Could not import from langchain_community or langchain_google_genai. Falling back to older imports (might not work with latest versions).")
    from langchain.document_loaders import IMSDbLoader
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains.question_answering import load_qa_chain
    from langchain.chains import LLMChain
    from langchain.schema import Document
    from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


app = Flask(__name__)

# --- Load Movie Titles from movies.txt ---
MOVIE_LIST = []
try:
    with open('movies.txt', 'r', encoding='utf-8') as f:
        MOVIE_LIST = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(MOVIE_LIST)} movie titles from movies.txt")
except FileNotFoundError:
    print("Error: movies.txt not found. Movie suggestions will not work.")
    MOVIE_LIST = [] # Ensure it's an empty list if file is missing
except Exception as e:
    print(f"Error loading movies.txt: {e}")
    MOVIE_LIST = []



# --- Langchain Setup ---
# IMPORTANT: Set your API key as an environment variable before running or deploying
# For Google Generative AI (Gemini), the variable name is typically GOOGLE_API_KEY
# Example: export GOOGLE_API_KEY='your-api-key' in your terminal
# Or configure it on your hosting provider's dashboard (e.g., Render, PythonAnywhere)
# Do NOT hardcode your API key here!
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = 'AIzaSyAy7dfDNT-gMXSRJYdqJY73H3cZKNDrAN0'
    print("Warning: GOOGLE_API_KEY environment variable not set.")
    print("Please set it before running the app or deployment will fail.")
    # As a fallback for local testing (REMOVE THIS IN PRODUCTION!):
    # os.environ["GOOGLE_API_KEY"] = "YOUR-LOCAL-TEST-KEY"


llm = None # Initialize to None
qa_chain = None # Initialize to None

try:
    # Use a suitable Gemini model for chat/QA

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    # We'll use the 'map_reduce' chain for question answering potentially long text (movie scripts)
    # This chain handles splitting the document if it's too long
    qa_chain = load_qa_chain(llm, chain_type="map_reduce")
except Exception as e:
    print(f"Error initializing LLM or QA chain: {e}")
    print("Ensure GOOGLE_API_KEY is set and valid.")

# --- Helper Function for Scraping Script ---
# Store scripts in memory to avoid re-scraping for the same movie in a session
# For a production app, consider a more robust caching mechanism (e.g., Redis, Flask-Caching)
SCRIPT_CACHE = {}


# --- Helper Function for Scraping Script ---
def scrape_script_given_name(movie_name):
    """
    Scrapes a movie script from IMSDb given the movie name.
    Handles potential errors and returns cleaned text or None.
    Caches the script to avoid re-scraping.
    """
    formatted_name_for_cache = movie_name.strip().lower()
    if formatted_name_for_cache in SCRIPT_CACHE:
        print(f"Using cached script for '{movie_name}'")
        return SCRIPT_CACHE[formatted_name_for_cache]

    # Format the movie name for the URL (IMSDb uses hyphens)
    # Take the first part before any '|' character if present (from your movies.txt format)
    core_movie_name = movie_name.split('|')[0].strip()
    # Basic cleaning for URL
    formatted_name_for_url = core_movie_name.replace(" ", "-").replace("'", "").replace(":", "").replace(",", "").replace("!", "").replace("?", "")
    url = f'https://imsdb.com/scripts/{formatted_name_for_url}.html'

    try:
        print(f"Attempting to scrape: {url}")
        loader = IMSDbLoader(url) # Pass the direct URL
        data = loader.load()

        if not data or not data[0].page_content:
            print(f"IMSDbLoader returned no data or empty content for {movie_name} ({url}).")
            SCRIPT_CACHE[formatted_name_for_cache] = None # Cache failure
            return None

        script_content = data[0].page_content
        cleaned_script = " ".join(script_content.split())
        script_document = Document(page_content=cleaned_script, metadata={"source": movie_name})
        
        SCRIPT_CACHE[formatted_name_for_cache] = [script_document] # Cache success
        return [script_document]

    except Exception as e:
        print(f"Error scraping script for '{movie_name}' from '{url}': {e}")
        SCRIPT_CACHE[formatted_name_for_cache] = None # Cache failure
        return None




# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main chat page."""
    return render_template('index.html')



@app.route('/ask', methods=['POST'])
def ask_movie_question():
    """
    Receives movie title and question, scrapes script (if not cached),
    asks question using Langchain/Gemini, and returns JSON response.
    """
    data = request.get_json() # Get data as JSON
    movie_title = data.get('movie_title', '').strip()
    user_question = data.get('user_question', '').strip()

    if not movie_title or not user_question:
        return jsonify({
            "error": "Movie title and question are required."
        }), 400

    if qa_chain is None or llm is None:
        print("Error: QA chain or LLM not initialized. Check GOOGLE_API_KEY and server logs.")
        return jsonify({
            "movie_title": movie_title,
            "user_question": user_question,
            "answer": "Error: AI question answering service not available. Please check server logs."
        }), 500

    print(f"Received request for movie: '{movie_title}', question: '{user_question}'")

    # 1. Get the script (from cache or scrape)
    script_documents = scrape_script_given_name(movie_title)

    if not script_documents: # This means scraping failed or returned None
        print(f"Script not found or failed to scrape for '{movie_title}'.")
        return jsonify({
            "movie_title": movie_title,
            "user_question": user_question,
            "answer": f"Could not find or scrape the script for '{movie_title}'. Please check the movie title and try again."
        })
    
    # Check if script content is substantial enough (e.g., more than a few hundred characters)
    # This is a basic check for empty or placeholder scripts.
    if len(script_documents[0].page_content) < 500: # Adjust threshold as needed
        print(f"Script for '{movie_title}' seems too short or is a placeholder. Length: {len(script_documents[0].page_content)}")
        return jsonify({
            "movie_title": movie_title,
            "user_question": user_question,
            "answer": f"The script found for '{movie_title}' appears to be incomplete or a placeholder. Please try a different movie or check the source."
        })


    print(f"Using script for '{movie_title}'. Length: {len(script_documents[0].page_content)} characters.")

    # 2. Use Langchain QA chain to answer the question
    try:
        prompt_string = f"""You are a helpful movie script expert. Your goal is to answer questions based ONLY on the provided movie script content. "
            Do not use any external knowledge. If the answer isn't in the script, say so. 
            Be concise and directly answer the user's question.
            If the user asks about a character, focus on their actions, dialogue, and descriptions within the script up to the point relevant to their query (if specified).
            If the user seems confused or asks for a recap, summarize key plot points and character involvements from the script relevant to their query.
            Avoid spoilers beyond what would be known at a certain point if the user indicates their viewing progress.
            Here is a movie script:
            # --- SCRIPT START ---
            # {script_documents[0].page_content}
            # --- SCRIPT END ---

            # Based on this script, please answer the following question:
            # Question: {user_question}
            #
            # Answer:"""

        answer = llm.invoke(prompt_string).content
        
        
        # If you want to use the more detailed ChatPromptTemplate with a "stuff" chain,
        # you might need to customize the underlying LLMChain's prompt within load_qa_chain,
        # or use a more direct approach with LCEL (LangChain Expression Language).

        # For now, keeping your original `qa_chain.run()` call:
        # The prompt you had before:
        #  prompt_for_llm = f"""You are a movie script expert. Answer the question below based on the script provided.
        # Question: {user_question}
        # If the user's question indicates approximately where they are in the movie, answer only with information up to that point.
        # Do not include any spoilers or information that occurs later in the movie.
        # If the user is confused be sure to include important plot points and characters up to that point in the movie.
        # If the user is asking about a character, be sure to include important plot points up to that point in the movie.
        # """
        # answer = qa_chain.run(input_documents=script_documents, question=prompt_for_llm)



        print(f"Successfully generated answer for '{movie_title}'.")

        return jsonify({
            "movie_title": movie_title,
            "user_question": user_question,
            "answer": answer
        })

    except Exception as e:
        print(f"Error during Langchain QA process for '{movie_title}': {e}")
        return jsonify({
            "movie_title": movie_title,
            "user_question": user_question,
            "answer": f"An error occurred while trying to answer your question using the script. Error: {e}"
        }), 500
    
# --- New Route for Suggestions ---
@app.route('/suggest_movies')
def suggest_movies():
    search_term = request.args.get('query', '').lower()
    suggestions = []

    if search_term and MOVIE_LIST:
        # Simple contains check and limit to 5
        count = 0
        for movie in MOVIE_LIST:
            if search_term in movie.lower():
                suggestions.append(movie)
                count += 1
                if count >= 5: # Limit to top 5
                    break

        # For more advanced similarity, you'd use a library like fuzzywuzzy here
        # from fuzzywuzzy import process
        # if search_term and MOVIE_LIST:
        #    # process.extract takes (query, choices, limit)
        #    # it returns a list of (match, score) tuples
        #    matches = process.extract(search_term, MOVIE_LIST, limit=5)
        #    suggestions = [match[0] for match in matches] # Just get the movie names

    return jsonify(suggestions) # Return suggestions as JSON


if __name__ == '__main__':
    # This is for local development only.
    # For deployment, Gunicorn or a similar server will run the app.
    # Make sure your GOOGLE_API_KEY environment variable is set!
    if os.environ.get("GOOGLE_API_KEY"):
        print("Running Flask app locally. Ensure GOOGLE_API_KEY is set.")
        # Use a different port if needed, debug=True restarts server on code changes
        app.run(debug=True, port=5000)
    else:
        print("\n!!!! Cannot run locally without GOOGLE_API_KEY environment variable set. !!!!\n")
        print("Please set it (e.g., export GOOGLE_API_KEY='your_key') and run again.")