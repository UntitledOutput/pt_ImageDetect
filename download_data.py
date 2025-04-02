import os
import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

import imghdr
import GoogleImageScraper as gimg

def download_google_images(query, num_images=10, output_folder="images"):
    urls = gimg.urls(query, limit=num_images)

    print(urls)

def download_data(path, imgs):

    download_google_images("puppies",num_images=5,output_folder=path+"/puppies/")

    for i in imgs:
        pass

    pass


import train
if __name__ == '__main__':
    download_data(train.datapath,train.classes)  # Call the train function to start training the model.
    pass
