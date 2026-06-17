import sys
import subprocess
from pathlib import Path

VIDEOS_DIR = Path("/Volumes/Crucial X8/LSM_Translator/videos/isolated")


def scrape_channel(channel_url: str) -> None:
    cmd = [
        "yt-dlp",
        channel_url,
        "--output", str(VIDEOS_DIR / "%(id)s/%(id)s, %(ext)s"),
        "--write-info-json",
        "--format", "mp4",
        "--ignore-errors",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 01_scrape.py <channel_url>")
        sys.exit(1)
    scrape_channel(sys.argv[1])
