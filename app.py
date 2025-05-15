import os
from flask import Flask, render_template, request, redirect, url_for, jsonify, session

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

app.secret_key = os.environ.get("FLASK_SECRET_KEY", "your_default_super_secret_key_for_dev")
if app.secret_key == "your_default_super_secret_key_for_dev" and os.environ.get("FLASK_ENV") != "development":
    print("Warning: Using default Flask secret key. Please set a strong FLASK_SECRET_KEY environment variable for production.")

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
    asks question using Langchain/Gemini with conversation history,
    and returns JSON response.
    """
    data = request.get_json()
    movie_title = data.get('movie_title', '').strip()
    user_question = data.get('user_question', '').strip()

    if not movie_title or not user_question:
        return jsonify({
            "error": "Movie title and question are required."
        }), 400

    if llm is None: # Simplified check as qa_chain is not directly used with llm.invoke for the main logic now
        print("Error: LLM not initialized. Check GOOGLE_API_KEY and server logs.")
        return jsonify({
            "movie_title": movie_title,
            "user_question": user_question,
            "answer": "Error: AI question answering service not available. Please check server logs."
        }), 500

    print(f"Received request for movie: '{movie_title}', question: '{user_question}'")

    script_documents = scrape_script_given_name(movie_title)

    if not script_documents:
        print(f"Script not found or failed to scrape for '{movie_title}'.")
        return jsonify({
            "movie_title": movie_title,
            "user_question": user_question,
            "answer": f"Could not find or scrape the script for '{movie_title}'. Please check the movie title and try again."
        })

    if len(script_documents[0].page_content) < 500:
        print(f"Script for '{movie_title}' seems too short. Length: {len(script_documents[0].page_content)}")
        return jsonify({
            "movie_title": movie_title,
            "user_question": user_question,
            "answer": f"The script found for '{movie_title}' appears to be incomplete. Please try a different movie."
        })

    print(f"Using script for '{movie_title}'. Length: {len(script_documents[0].page_content)} characters.")

    try:
        # Retrieve chat history for the current movie from session
        session_history_key = f"chat_history_{movie_title.replace(' ', '_').lower()}" # Make key session-friendly
        chat_history = session.get(session_history_key, [])

        formatted_history = ""
        if chat_history:
            formatted_history += "Context from our previous conversation about this movie:\n"
            for entry in chat_history:
                formatted_history += f"Your previous question: {entry['user']}\nMy previous answer: {entry['ai']}\n\n"
            formatted_history += "---\n"

        prompt_string = f"""You are a movie expert. Your goal is to answer questions based on the provided movie script content AND the conversation history if available AND what you know about the movie.
        You should be very cautious about making up information or spoiling the movie.
        If the user refers to previous parts of our conversation, use that context intelligently.
        Do not be afraid to ask clarifying questions if the user's question is vague or could be interpreted in multiple ways.
        You are allowed to reason about the question. You can provide some opinions about why that scene is important, what the director was trying to convey, etc.

        Here is the movie script:
        --- SCRIPT START ---
        {script_documents[0].page_content}
        --- SCRIPT END ---

        {formatted_history}
        
        Based on the script AND the conversation history (if any), please answer the following new question about the movie "{movie_title}":
        New Question: {user_question}

        Answer:"""

        # Make sure your LLM can handle the combined length of script + history + question.
        # If it's too long, you might need to truncate the script or history.
        ai_response = llm.invoke(prompt_string).content

        # Update chat history in session
        # Limit history length to prevent excessively large sessions/prompts
        max_history_length = 10 # Keep last 5 Q&A pairs (10 entries)
        chat_history.append({"user": user_question, "ai": ai_response})
        if len(chat_history) > max_history_length:
            chat_history = chat_history[-max_history_length:]
        session[session_history_key] = chat_history
        session.modified = True # Ensure session is saved

        print(f"Successfully generated answer for '{movie_title}'.")

        return jsonify({
            "movie_title": movie_title,
            "user_question": user_question,
            "answer": ai_response
        })

    except Exception as e:
        print(f"Error during Langchain QA process for '{movie_title}': {e}")
        # Log the full error for debugging, but return a generic message to the user.
        import traceback
        traceback.print_exc()
        return jsonify({
            "movie_title": movie_title,
            "user_question": user_question,
            "answer": f"An error occurred while trying to answer your question. Please try again. Error details: {str(e)}"
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


@app.route('/clear_chat_history', methods=['POST'])
def clear_chat_history_route():
    data = request.get_json()
    movie_title_to_clear = data.get('movie_title', '').strip()

    if not movie_title_to_clear:
        return jsonify({"error": "Movie title is required to clear history."}), 400

    session_history_key = f"chat_history_{movie_title_to_clear.replace(' ', '_').lower()}"
    if session_history_key in session:
        del session[session_history_key]
        session.modified = True # Ensure session is saved after deletion
        print(f"Chat history cleared for movie: '{movie_title_to_clear}'")
        return jsonify({"message": f"Chat history for '{movie_title_to_clear}' cleared successfully."}), 200
    else:
        print(f"No chat history found in session for movie: '{movie_title_to_clear}'")
        return jsonify({"message": f"No chat history to clear for '{movie_title_to_clear}'."}), 200

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