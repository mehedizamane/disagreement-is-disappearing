!pip install feedparser
import feedparser
import requests
import os
import re
from datetime import datetime

# RSS feed URL
rss_feed_url = "https://feeds.megaphone.fm/BMDC3567910388"

# Parse the RSS feed
feed = feedparser.parse(rss_feed_url)

# Directory to save downloaded episodes
output_dir = "podcast_episodes"
os.makedirs(output_dir, exist_ok=True)

# Date range: January 1, 2024 to December 31, 2024
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 1, 3)

# Function to sanitize filenames
def sanitize_filename(name):
    """Remove invalid characters from filename."""
    return re.sub(r'[^A-Za-z0-9_\-]', '_', name)

# Download episodes published between January 2020 and December 2020
for entry in feed.entries:
    published_date = datetime(*entry.published_parsed[:6])  # Convert to datetime object
    if start_date <= published_date <= end_date:
        # Sanitize episode title and format the filename
        episode_title = sanitize_filename(entry.title[:20].strip())
        date_str = published_date.strftime("%Y-%m-%d")
        filename = f"{date_str}_{episode_title}.mp3"
        
        audio_url = entry.enclosures[0].href  # Get audio URL from the enclosure
        audio_path = os.path.join(output_dir, filename)
        
        print(f"Downloading: {entry.title} (Published on {published_date})")
        response = requests.get(audio_url)
        
        with open(audio_path, "wb") as f:
            f.write(response.content)

print("All episodes from 2023-Jan downloaded.")
