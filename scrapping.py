from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time, os
import pandas as pd

url = input("Masukkan url toko : ")

if url:
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=options)
    driver.get(url)

    data = []
    for i in range(0, 10):
        soup = BeautifulSoup(driver.page_source, "html.parser")
        containers = soup.findAll('article', attrs={'class': 'css-ccpe8t'})

        for container in containers:
            try:
                review = container.find('span', attrs={'data-testid': 'lblItemUlasan'}).text
                rating = container.find('div', attrs={'data-testid': 'icnStarRating'})['aria-label'] if container.find('div', attrs={'data-testid': 'icnStarRating'}) else "N/A"

                rating_mapping = {"bintang 1": 1, "bintang 2": 2, "bintang 3": 3, "bintang 4": 4, "bintang 5": 5}
                rating = rating_mapping.get(rating, "N/A")

                data.append(
                    (review, rating)
                )
            except AttributeError:
                continue

        time.sleep(2)
        driver.find_element(By.CSS_SELECTOR, "button[aria-label^='Laman berikutnya']").click()
        time.sleep(3)

    print(data)
    df = pd.DataFrame(data, columns=["Ulasan", "Rating"])

    downloads_folder_path = os.path.join(os.path.expanduser("~"), "Downloads")
    csv_file_path = os.path.join(downloads_folder_path, "Tokopedia.csv")

    df.to_csv(csv_file_path, index=False)
