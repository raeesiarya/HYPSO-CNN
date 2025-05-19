import os
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Angi Chrome-binærens sti
chrome_path = "/usr/bin/google-chrome"

# Konfigurer Chrome-alternativer
chrome_options = Options()
chrome_options.binary_location = chrome_path
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--headless")  # Valgfritt: Kjør uten GUI

# Start Chrome WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# Hoved-URL
base_url = "http://129.241.2.147:8009/"

# Opprett hovedmappen "collected_data" hvis den ikke finnes
base_save_dir = "raw_data"
os.makedirs(base_save_dir, exist_ok=True)

# 1️⃣ Hent directories fra hovedsiden
driver.get(base_url)
soup = BeautifulSoup(driver.page_source, "html.parser")
directories = [a.text.strip() for a in soup.find_all("a")]

# 2️⃣ Gå inn i første nivå undermapper (hoved-mapper)
for directory in directories:
    first_level_url = base_url + directory
    driver.get(first_level_url)
    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Finn undermapper inne i hovedmappen
    subdirs = [a.text.strip() for a in soup.find_all("a")]

    # 3️⃣ Gå inn i undermapper for å hente scale3.png og organisere captchures
    for subdir in subdirs:
        second_level_url = first_level_url + subdir
        driver.get(second_level_url)
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Opprett en mappe for hver captchure
        captchure_dir = os.path.join(base_save_dir, directory.strip("/"), subdir.strip("/"))
        os.makedirs(captchure_dir, exist_ok=True)

        # Hent scale3.png-filer
        scale3_files = [a.text.strip() for a in soup.find_all("a") if ".png" in a.text.strip().lower()]
        for file in scale3_files:
            file_url = second_level_url + file
            save_path = os.path.join(captchure_dir, file)
            response = requests.get(file_url)
            with open(save_path, "wb") as f:
                f.write(response.content)
                print(f"Lastet ned: {file}")
                print("---")

        # Hent .bip-filer og legg dem i samme mappe
        bip_files = [a.text.strip() for a in soup.find_all("a") if a.text.strip().lower().endswith(".bip") or a.text.strip().lower().endswith(".bip@")]
        for file in bip_files:
            file_url = second_level_url + file[:-1] if file.endswith("@") else second_level_url + file
            save_path = os.path.join(captchure_dir, file)
            response = requests.get(file_url)
            with open(save_path, "wb") as f:
                f.write(response.content)
                print(f"Lastet ned: {file}")
                print("---")

# Lukk nettleseren
driver.quit()

print("\n✅ Alle scale3.png- og .bip-filer er lastet ned og organisert i hver sin captchure-mappe!")