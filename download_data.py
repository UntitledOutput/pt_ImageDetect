import os
import time
from duckduckgo_search import DDGS
import requests
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from fake_useragent import UserAgent
import random
from PIL import Image
#import traceback


ua = UserAgent()

def download_image(url, folder, index):
    try:

        headers = {"User-Agent": ua.random}
        response = requests.get(url, stream=True, timeout=10,headers=headers)
        response.raise_for_status()

        file_path = os.path.join(folder, f"image_{index+1}.jpg")
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)

        print(f"Downloaded: {file_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")


def download_images_multithreaded(image_urls, folder):
    
    # Use ThreadPoolExecutor to download multiple images at once
    with ThreadPoolExecutor(max_workers=10) as executor:
        for index, url in enumerate(image_urls):
            executor.submit(download_image, url, random.choice(folder), index)

# Function to get DuckDuckGo image URLs
def duckduckgo_image_search(query, num_images=10):
    with DDGS() as ddgs:
        results = list(ddgs.images(query, max_results=num_images,type_image="photo", layout="square",))
    return [result["image"] for result in results]

def check_images(folder):
    for file in os.listdir(folder):
        try:
            img_path = os.path.join(folder, file)
            img = Image.open(img_path)
            img.verify()  # Verify that it is an image

            if (img.mode not in ["RGB"]):
                img = Image.open(img_path)
                img = img.convert('RGB')  # Convert to RGB to handle different formats
                img.save(img_path, "JPEG")  # Save it back to ensure it's a valid JPEG

            img.close()  # Close the image file

        except FileNotFoundError:
            print(f"File not found: {img_path}")
            continue
        except Exception as e:
            #print(traceback.format_exc())
            print(f"\033[91mError processing {img_path}: {e}\033[0m")
            print(f"\tRemoved corrupted image: {img_path}")

            img.close()  # Close the image file if it was opened

            os.remove(img_path)
            continue

def check_data_images(folder):

        for subfolder in ["train", "test"]:
            subfolder_path = os.path.join(folder, subfolder)
            if not os.path.exists(subfolder_path):
                print(f"Warning: {subfolder_path} does not exist.")
                continue

            for cls in os.listdir(subfolder_path):
                cls_path = os.path.join(subfolder_path, cls)
                if not os.path.isdir(cls_path):
                    print(f"Warning: {cls_path} is not a directory.")
                    continue

                check_images(cls_path)
        print("\033[92mAll images checked and cleaned successfully.\033[0m")

def download_google_images(query, number_of_images=10, output_folder="images"):
    image_urls = duckduckgo_image_search(query=query, num_images=number_of_images)

    output_folder1 = output_folder + "train/" + query.replace(" ", "_").lower()
    output_folder2 = output_folder + "test/" + query.replace(" ", "_").lower()


    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder1, exist_ok=True)
    os.makedirs(output_folder2, exist_ok=True)

    download_images_multithreaded(image_urls, folder=[output_folder1,output_folder2])

    

def download_data(path, imgs):

    img_count = 200

    for i in imgs:
        download_google_images(i,number_of_images=img_count,output_folder=path)
        pass

    check_data_images(path)  # Check and clean images in train/test folders

    pass


import train
if __name__ == '__main__':
    download_data(train.datapath,train.classes)  # Call the train function to start training the model.
    pass
