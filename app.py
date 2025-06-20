import os
import time
import re
from datetime import datetime, date
import psycopg2
from psycopg2 import sql
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
API_KEY = os.getenv("API_KEY")

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

YOUTUBE_API_SERVICE_NAME = os.getenv("YOUTUBE_API_SERVICE_NAME")
YOUTUBE_API_VERSION = os.getenv("YOUTUBE_API_VERSION")

def get_youtube_service():
    """Builds and returns a YouTube API service object."""
    try:
        return build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
    except Exception as e:
        print(f"Error building YouTube service: {e}")
        print("Please ensure your API_KEY is correct and the YouTube Data API v3 is enabled for your project.")
        return None

def get_db_connection():
    """Establishes and returns a PostgreSQL database connection."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        print("Successfully connected to the database.")
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        print("Please check your database credentials and ensure PostgreSQL is running.")
        return None

def parse_duration_to_seconds(duration_iso8601):
    """
    Parses an ISO 8601 duration string (e.g., 'PT1H2M3S') into total seconds.
    The YouTube API returns durations in this format.
    """

    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_iso8601)
    if not match:
        return 0 # Return 0 for invalid or unparseable durations

    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    seconds = int(match.group(3)) if match.group(3) else 0

    total_seconds = (hours * 3600) + (minutes * 60) + seconds
    return total_seconds

def get_channel_uploads_playlist_id(youtube_service, channel_id):
    """
    Fetches the uploads playlist ID for a given channel ID.
    This playlist contains all public videos uploaded by the channel.
    """
    if not youtube_service:
        return None
    try:
        channels_response = youtube_service.channels().list(
            id=channel_id,
            part='contentDetails'
        ).execute()

        if channels_response and 'items' in channels_response and len(channels_response['items']) > 0:
            return channels_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        else:
            print(f"Could not find uploads playlist for channel ID: {channel_id}")
            return None
    except HttpError as e:
        print(f"HTTP error fetching uploads playlist for {channel_id}: {e}")
        if e.resp.status == 403:
            print("Quota exceeded or API key issue.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred getting uploads playlist for {channel_id}: {e}")
        return None

def video_exists_in_db(cursor, video_id, channel_id):
    """Checks if a video already exists in the 'video' table for a given channel."""
    try:
        cursor.execute(
            sql.SQL("SELECT 1 FROM video WHERE id = %s AND channel_id = %s"),
            (video_id, channel_id)
        )
        return cursor.fetchone() is not None
    except psycopg2.Error as e:
        print(f"Error checking if video {video_id} exists in DB: {e}")
        return True # Assume it exists to avoid duplicates in case of error

def insert_video_data(cursor, video_data):
    """Inserts a single video's data into the 'video' table."""
    try:
        insert_query = sql.SQL("""
            INSERT INTO video (
                id, channel_id, title, category_id, tags, published_at,
                view_count, like_count, comment_count, duration, video_url, thumbnail_url
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING; -- Avoids inserting duplicates based on video ID
        """)
        cursor.execute(insert_query, (
            video_data['id'],
            video_data['channelId'],
            video_data['title'],
            video_data['categoryId'],
            video_data['tags'],
            video_data['publishedAt'],
            video_data['viewCount'],
            video_data['likeCount'],
            video_data['commentCount'],
            video_data['duration'],
            video_data['video_url'],
            video_data['thumbnail_url']
        ))
        return True
    except psycopg2.Error as e:
        print(f"Error inserting video {video_data.get('id', 'N/A')} into DB: {e}")
        return False

def update_channel_collection_status(cursor, channel_id, status=True):
    """Updates the videos_collected flag for a channel."""
    try:
        update_query = sql.SQL("""
            UPDATE channel
            SET videos_collected = %s
            WHERE id = %s;
        """)
        cursor.execute(update_query, (status, channel_id))
        return True
    except psycopg2.Error as e:
        print(f"Error updating channel {channel_id} collection status: {e}")
        return False

def collect_videos_for_channel(youtube_service, db_conn, channel_id, min_publish_date_str="2020-01-01"):
    """
    Collects video data for a given channel from its uploads playlist
    and inserts it into the database. Filters by min_publish_date.
    """
    print(f"\n--- Starting video collection for channel ID: {channel_id} ---")
    min_publish_date = datetime.strptime(min_publish_date_str, '%Y-%m-%d').date()
    cursor = db_conn.cursor()
    videos_collected_count = 0

    # 1. Get uploads playlist ID
    uploads_playlist_id = get_channel_uploads_playlist_id(youtube_service, channel_id)
    if not uploads_playlist_id:
        print(f"Skipping channel {channel_id}: Could not get uploads playlist ID.")
        return False

    next_page_token = None
    keep_fetching = True

    while keep_fetching:
        try:
            # 2. Fetch videos from the uploads playlist (max 50 per request)
            playlist_items_response = youtube_service.playlistItems().list(
                playlistId=uploads_playlist_id,
                part='snippet,contentDetails',
                maxResults=50,
                pageToken=next_page_token
            ).execute()

            items = playlist_items_response.get('items', [])
            if not items:
                print("No more videos in playlist or reached end.")
                keep_fetching = False
                break

            video_ids_to_fetch_details = []
            video_data_buffer = {}

            for item in items:
                video_id = item['snippet']['resourceId']['videoId']
                published_at_str = item['snippet']['publishedAt']
                published_date = datetime.strptime(published_at_str, '%Y-%m-%dT%H:%M:%SZ').date()

                if published_date < min_publish_date:
                    print(f"Reached video older than {min_publish_date_str}. Stopping for this channel.")
                    keep_fetching = False
                    break # Stop fetching for this channel

                # If video is already in DB, skip it (handles resuming)
                if video_exists_in_db(cursor, video_id, channel_id):
                    continue

                # Buffer basic video info (duration will be updated from videos.list call)
                video_data_buffer[video_id] = {
                    'id': video_id,
                    'channelId': channel_id,
                    'title': item['snippet']['title'],
                    'categoryId': item['snippet'].get('categoryId', 'N/A'),
                    'tags': item['snippet'].get('tags', []),
                    'publishedAt': published_date,
                    'video_url': f"https://www.youtube.com/watch?v={video_id}",
                    'thumbnail_url': item['snippet']['thumbnails'].get('high', {}).get('url') or \
                                     item['snippet']['thumbnails'].get('medium', {}).get('url') or \
                                     item['snippet']['thumbnails'].get('default', {}).get('url') or \
                                     'N/A',
                    'duration': 0, # Initialize duration to 0, will be updated in next API call
                    'viewCount': 0, # Initialize counts, will be updated in next API call
                    'likeCount': 0,
                    'commentCount': 0
                }
                video_ids_to_fetch_details.append(video_id)

            # If broke early because of date, or all videos were skipped/existing, break outer loop
            if not keep_fetching and (not items or published_date < min_publish_date):
                break

            # 3. Fetch full video details (including statistics AND contentDetails for duration) in a batch
            if video_ids_to_fetch_details:
                # The API allows up to 50 IDs per request for videos.list
                for j in range(0, len(video_ids_to_fetch_details), 50):
                    batch_video_ids = video_ids_to_fetch_details[j:j + 50]
                    try:
                        videos_response = youtube_service.videos().list(
                            id=','.join(batch_video_ids),
                            part='statistics,contentDetails,snippet'
                        ).execute()

                        for video_item in videos_response.get('items', []):
                            vid_id = video_item['id']
                            # Update buffer with statistics, duration, categoryId, and tags
                            if vid_id in video_data_buffer:
                                stats = video_item.get('statistics', {})
                                content_details = video_item.get('contentDetails', {})
                                snippet = video_item.get('snippet', {}) # Get snippet from videos.list response

                                video_data_buffer[vid_id]['viewCount'] = int(stats.get('viewCount', 0))
                                video_data_buffer[vid_id]['likeCount'] = int(stats.get('likeCount', 0))
                                video_data_buffer[vid_id]['commentCount'] = int(stats.get('commentCount', 0))

                                duration_str = content_details.get('duration', 'PT0S')
                                video_data_buffer[vid_id]['duration'] = parse_duration_to_seconds(duration_str)

                                # Explicitly get categoryId and tags from the videos.list snippet
                                video_data_buffer[vid_id]['categoryId'] = snippet.get('categoryId', 'N/A')
                                video_data_buffer[vid_id]['tags'] = snippet.get('tags', [])

                                print(f"  Extracted Category ID: {video_data_buffer[vid_id]['categoryId']}")
                                print(f"  Extracted Tags: {video_data_buffer[vid_id]['tags']}")
                                print(f"  Extracted Duration: {video_data_buffer[vid_id]['duration']} seconds")

                                # 4. Insert into database
                                if insert_video_data(cursor, video_data_buffer[vid_id]):
                                    videos_collected_count += 1
                                    if videos_collected_count % 10 == 0:
                                        db_conn.commit() # Commit every 10 videos to save progress
                                        print(f"Collected {videos_collected_count} videos for channel {channel_id}")

                        time.sleep(0.1) # Small delay after each batch of stats fetch

                    except HttpError as e:
                        print(f"HTTP error fetching video statistics/contentDetails/snippet for batch {j//50 + 1}: {e}")
                        db_conn.rollback() # Rollback current transaction on API error
                        raise # Re-raise to stop processing this channel if API issue persists
                    except Exception as e:
                        print(f"An unexpected error occurred during video stats/contentDetails/snippet fetch: {e}")
                        db_conn.rollback()
                        raise

            next_page_token = playlist_items_response.get('nextPageToken')
            if not next_page_token:
                keep_fetching = False # No more pages

            time.sleep(0.5) # Delay between pagination requests to avoid rate limits

        except HttpError as e:
            print(f"HTTP error during playlist item fetching for channel {channel_id}: {e}")
            db_conn.rollback()
            if e.resp.status == 403:
                print("Quota exceeded or API key issue. Stopping collection for this channel.")
            return False # Indicate failure for this channel
        except Exception as e:
            print(f"An unexpected error occurred during playlist item fetching for channel {channel_id}: {e}")
            db_conn.rollback()
            return False

    db_conn.commit() # Final commit for the channel's videos
    print(f"Finished collecting {videos_collected_count} new videos for channel ID: {channel_id}.")
    return True # Indicate success for this channel

def main():
    # --- Initialize YouTube service ---
    youtube = get_youtube_service()
    if not youtube:
        return

    # --- Establish Database Connection ---
    db_conn = get_db_connection()
    if not db_conn:
        return

    try:
        cursor = db_conn.cursor()

        # 1. Get channel IDs from 'channel' table that haven't been processed
        print("Fetching channels from database...")
        cursor.execute(
            sql.SQL("SELECT id FROM channel WHERE videos_collected = FALSE")
        )
        channels_to_process = [row[0] for row in cursor.fetchall()]
        print(f"Found {len(channels_to_process)} channels to process.")

        if not channels_to_process:
            print("All channels already processed or no channels found to process.")
            return

        # 2. Iterate through each channel and collect videos
        for i, channel_id in enumerate(channels_to_process):
            print(f"\n--- Processing channel {i+1}/{len(channels_to_process)}: {channel_id} ---")
            if collect_videos_for_channel(youtube, db_conn, channel_id, min_publish_date_str="2020-01-01"):
                # If successful, update the channel's status in DB
                if update_channel_collection_status(cursor, channel_id):
                    db_conn.commit() # Commit the status update
                    print(f"Successfully updated collection status for channel: {channel_id}")
                else:
                    db_conn.rollback() # Rollback status update if it failed
                    print(f"Failed to update collection status for channel: {channel_id}")
            else:
                db_conn.rollback() # Rollback any partial inserts for this channel if collection failed
                print(f"Failed or interrupted collection for channel: {channel_id}. Data for this channel may be incomplete.")

    except Exception as e:
        print(f"An error occurred in the main process: {e}")
        db_conn.rollback() # Ensure any open transaction is rolled back
    finally:
        if db_conn:
            db_conn.close()
            print("Database connection closed.")

if __name__ == '__main__':
    main()
