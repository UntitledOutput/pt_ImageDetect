import os
import google_images_download.google_images_download as gid

def download_data(path, imgs):


    for i in imgs:
        response = gid.googleimagesdownload()

        arguments = {
            "keywords":i,
            "limit":20,
            "print_urls":False
        }

        paths = response.download(arguments)
        print(paths)

    pass