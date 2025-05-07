api_key = 'AIzaSyAy7dfDNT-gMXSRJYdqJY73H3cZKNDrAN0'

import requests

url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
headers = {"Content-Type": "application/json"}
data = {
    "contents": [
        {
            "parts": [{"text": "Explain how AI works"}]
        }
    ]
}
