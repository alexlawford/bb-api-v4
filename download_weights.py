import os
import requests

def download_file(url, file_name, directory):
    """
    Download a file from the given URL and save it into the specified directory.
    
    Parameters:
        url (str): The URL of the file to download.
        directory (str): The directory where the file will be saved.
        
    Returns:
        str: The path to the downloaded file.
    """

    # Make dir if it doesn't exist
    os.makedirs(directory, exist_ok=True)
        
    # Path
    save_path = os.path.join(directory, file_name)
    
    # Download
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    return save_path

# Download and save files
files = [
    ('https://boardsbot.b-cdn.net/inkSketch_V1.5.safetensors', 'inkSketch_V1.5.safetensors'),
    ('https://boardsbot.b-cdn.net/dreamshaper_8.safetensors', 'dreamshaper_8.safetensors'),
]

print("Downloading\n")

for url, file_name in files:
    downloaded_file_path = download_file(url, file_name, 'weights')
    print(f"File downloaded and saved to: {downloaded_file_path}")