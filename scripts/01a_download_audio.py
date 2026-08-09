"""
Phase 2: Audio Acquisition Pipeline
Downloads audio from YouTube videos using yt-dlp and formats them 
as 16kHz mono WAV files (required for Whisper ASR).
"""

import os
import argparse
import yt_dlp
import pandas as pd

def download_audio(urls, label, output_dir="data/raw_audio"):
    """
    Downloads audio from a list of URLs and saves them as 16kHz Mono WAV.
    Returns a list of dictionaries with metadata for the dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = []
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        # Whisper requires 16000Hz, 1 channel (mono)
        'postprocessor_args': [
            '-ar', '16000',
            '-ac', '1'
        ],
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'quiet': False,
        'extract_flat': False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in urls:
            try:
                print(f"\n--- Downloading: {url} ---")
                info_dict = ydl.extract_info(url, download=True)
                video_id = info_dict.get("id", None)
                
                if video_id:
                    file_path = os.path.join(output_dir, f"{video_id}.wav")
                    metadata.append({
                        "video_id": video_id,
                        "url": url,
                        "file_path": file_path,
                        "label": label,
                        "duration": info_dict.get("duration", 0)
                    })
                    print(f"Successfully saved {file_path}")
            except Exception as e:
                print(f"Failed to download {url}: {e}")
                
    return metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_run", action="store_true", help="Only download 1 short video for testing")
    args = parser.parse_args()

    # Curated URLs
    # Scams (Label 1) - E.g., Kitboga, Scammer Payback
    scam_urls = [
        "https://www.youtube.com/watch?v=J5z4VsWNEWE", # Kitboga - 10 min 
        "https://www.youtube.com/watch?v=XQZ9B_F-Gkc"  # Scammer Payback
    ]
    
    # Legit (Label 0) - Conversational podcasts, tech support, etc.
    legit_urls = [
        "https://www.youtube.com/watch?v=R23EifP_H00", # Customer Service Roleplay
        "https://www.youtube.com/watch?v=fW4V9xLq5QY"  # Tech support tutorial
    ]
    
    if args.test_run:
        print("TEST RUN: Downloading only 1 short scam video and 1 short legit video.")
        scam_urls = [scam_urls[0]]
        legit_urls = [legit_urls[0]]
        
    print("Initializing Phase 2 Audio Acquisition...")
    
    scam_meta = download_audio(scam_urls, label=1)
    legit_meta = download_audio(legit_urls, label=0)
    
    # Strict 1:1 Balancing Check based on Duration
    total_scam_time = sum(m["duration"] for m in scam_meta)
    total_legit_time = sum(m["duration"] for m in legit_meta)
    
    print("\n--- Download Summary ---")
    print(f"Scam Audio: {len(scam_meta)} files, {total_scam_time/60:.2f} total minutes")
    print(f"Legit Audio: {len(legit_meta)} files, {total_legit_time/60:.2f} total minutes")
    
    if abs(total_scam_time - total_legit_time) > 300: # 5 minute warning threshold
        print("WARNING: Audio duration is highly imbalanced! In a production run, please ensure scam and legit audio durations are roughly equal.")
    else:
        print("SUCCESS: Audio duration is perfectly balanced (1:1 ratio).")
        
    # Save metadata index
    df = pd.DataFrame(scam_meta + legit_meta)
    meta_path = "data/raw_audio/audio_metadata.csv"
    df.to_csv(meta_path, index=False)
    print(f"\nMetadata saved to {meta_path}")

if __name__ == "__main__":
    main()
