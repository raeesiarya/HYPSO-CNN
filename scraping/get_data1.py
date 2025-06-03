import os
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

chrome_path = "/usr/bin/google-chrome"

chrome_options = Options()
chrome_options.binary_location = chrome_path
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--headless")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

def scrape():  
    """
    Scrapes the specified website for directories and files, downloading images and BIP files.
    """

    base_url = "http://129.241.2.147:8009/"

    base_save_dir = "raw_data"
    os.makedirs(base_save_dir, exist_ok=True)

    driver.get(base_url)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    directories = [a.text.strip() for a in soup.find_all("a")]

    for directory in directories:
        first_level_url = base_url + directory
        driver.get(first_level_url)
        soup = BeautifulSoup(driver.page_source, "html.parser")

        subdirs = [a.text.strip() for a in soup.find_all("a")]

        for subdir in subdirs:
            second_level_url = first_level_url + subdir
            driver.get(second_level_url)
            soup = BeautifulSoup(driver.page_source, "html.parser")

            captchure_dir = os.path.join(base_save_dir, directory.strip("/"), subdir.strip("/"))
            os.makedirs(captchure_dir, exist_ok=True)

            scale3_files = [a.text.strip() for a in soup.find_all("a") if ".png" in a.text.strip().lower()]
            for file in scale3_files:
                file_url = second_level_url + file
                save_path = os.path.join(captchure_dir, file)
                response = requests.get(file_url)
                with open(save_path, "wb") as f:
                    f.write(response.content)
                    print(f"Downloaded: {file}")
                    print("---")

            bip_files = [a.text.strip() for a in soup.find_all("a") if a.text.strip().lower().endswith(".bip") or a.text.strip().lower().endswith(".bip@")]
            for file in bip_files:
                file_url = second_level_url + file[:-1] if file.endswith("@") else second_level_url + file
                save_path = os.path.join(captchure_dir, file)
                response = requests.get(file_url)
                with open(save_path, "wb") as f:
                    f.write(response.content)
                    print(f"Downloaded: {file}")
                    print("---")

    driver.quit()

    print("\n Scraping complete!")

scrape()