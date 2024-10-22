from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import requests
import os
from bs4 import BeautifulSoup
from PIL import Image
import io
import time

options = Options()
options.binary_location = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"

driver = webdriver.Chrome(options=options)

# Navigate to the website
driver.get("https://pb.nalog.ru/search.html#t=1728398478302&mode=search-all&queryAll=7709979413&page=1&pageSize=10")

# Switch to the iframe
iframe = WebDriverWait(driver, 10).until(
    EC.frame_to_be_available_and_switch_to_it((By.XPATH, "//iframe"))
)


image_counter = 1

while True:
    # # Wait for the button to be clickable
    # button_element = WebDriverWait(driver, 10).until(
    #     EC.element_to_be_clickable((By.XPATH, "//a[@href='#' and @tabindex='1000' and @class='ml-3']"))
    # )

    # # Add a 10-second delay before clicking the button
    # time.sleep(0.2)

    # # Click on the button
    # button_element.click()

    # Get the HTML content
    html = driver.page_source

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')

    # Find the image element
    image_element = soup.select_one('img[alt*="Необходимо включить загрузку картинок в браузере"]')

    # Get the image URL
    image_url = image_element.get('src')
    print(image_url)

    captcha_image_url = f"https://pb.nalog.ru{image_url}"
    captcha_image = requests.get(captcha_image_url)
    image = Image.open(io.BytesIO(captcha_image.content))
    image.save(f"C:\Captcha1\kriat.datoset\pik_1000\{image_counter}.jpg", format="JPEG")

    # Increment the image counter
    image_counter += 1
    if image_counter == 1001:
        break

