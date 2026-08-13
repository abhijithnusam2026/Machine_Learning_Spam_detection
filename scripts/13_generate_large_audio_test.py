import os
import pandas as pd
import random
import subprocess
import asyncio
from pathlib import Path

# Try to import edge_tts, if not found, we use it via subprocess
try:
    import edge_tts
except ImportError:
    import sys
    print("Installing edge-tts...")
    subprocess.run([sys.executable, "-m", "pip", "install", "edge-tts"])
    import edge_tts

async def generate_audio(text, output_mp3, voice="en-US-ChristopherNeural"):
    """Generate audio from text using edge-tts"""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_mp3)

def convert_to_wav(input_mp3, output_wav):
    """Convert MP3 to 16kHz WAV using FFmpeg"""
    subprocess.run([
        "ffmpeg", "-y", "-i", input_mp3, 
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_wav
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

async def main():
    print("--- Generating Large-Scale Audio Holdout Dataset ---")
    
    test_csv = "data/phase1.5/test.csv"
    if not os.path.exists(test_csv):
        print(f"Error: {test_csv} not found.")
        return
        
    df = pd.read_csv(test_csv)
    
    # Filter for long enough texts
    df = df[df['text'].str.len() > 50]
    
    # Get 50 scam and 50 legit
    scam_df = df[df['label'] == 1].sample(50, random_state=42)
    legit_df = df[df['label'] == 0].sample(50, random_state=42)
    
    final_df = pd.concat([scam_df, legit_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    out_dir = "data/large_audio_test"
    os.makedirs(out_dir, exist_ok=True)
    
    # Voices to alternate
    voices = ["en-US-ChristopherNeural", "en-US-JennyNeural", "en-US-GuyNeural", "en-US-AriaNeural"]
    
    print(f"Synthesizing {len(final_df)} audio files from TTS...")
    
    manifest = []
    
    for i, row in final_df.iterrows():
        text = str(row['text']).strip()
        label = row['label']
        
        # Limit text to ~200 chars for TTS generation speed
        if len(text) > 200:
            text = text[:200] + "..."
            
        voice = random.choice(voices)
        mp3_path = os.path.join(out_dir, f"sample_{i}.mp3")
        wav_path = os.path.join(out_dir, f"sample_{i}_class_{label}.wav")
        
        print(f"[{i+1}/{len(final_df)}] Generating audio for class {label} (Voice: {voice})")
        await generate_audio(text, mp3_path, voice=voice)
        convert_to_wav(mp3_path, wav_path)
        
        os.remove(mp3_path) # Cleanup mp3
        
        manifest.append({
            "file": wav_path,
            "label": label,
            "original_text": text
        })
        
    manifest_df = pd.DataFrame(manifest)
    manifest_csv = os.path.join(out_dir, "manifest.csv")
    manifest_df.to_csv(manifest_csv, index=False)
    
    print(f"\nSuccessfully generated {len(manifest)} holdout audio files in {out_dir}")
    print("Uploading to DagsHub S3...")
    
    from dotenv import load_dotenv
    load_dotenv()
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    token = os.getenv("MLFLOW_TRACKING_PASSWORD")
    
    if repo_owner and repo_name and token:
        import dagshub
        dagshub.auth.add_app_token(token)
        s3 = dagshub.get_repo_bucket_client(f"{repo_owner}/{repo_name}")
        
        for f in os.listdir(out_dir):
            local_path = os.path.join(out_dir, f)
            remote_path = f"artifacts/feature/phase-3.5-benchmark/large_audio_test/{f}"
            print(f"Uploading {f} to {remote_path}...")
            s3.upload_file(local_path, repo_name, remote_path)
            
        print("Upload complete!")

if __name__ == "__main__":
    asyncio.run(main())
