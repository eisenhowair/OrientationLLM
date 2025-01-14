from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import json

from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

diplomas_urls = ["b-u-t", 
            "licences", 
            "licences-professionnelles", 
            "masters", 
            "antenne-de-l-ecole-doctorale",
            "diplomes-d-ingenieurs-cycles-preparatoires",
            "diplomes-d-universite"]


def scrape_formation_links(driver, url):
    """
    Retrieve all formation links with pagination
    """
    driver.get(url)
    time.sleep(1)

    wait = WebDriverWait(driver, 1)
    try:
        cookie_banner = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#cookie-banner')))
        close_button = cookie_banner.find_element(By.CSS_SELECTOR, '.cookiebanner-refuse')  # Ajuste le sélecteur selon le bouton de fermeture
        close_button.click()
    except Exception as e:
        print("None cookies banner.")

    hrefs = []
    while True:
        a_elements = driver.find_elements(By.CSS_SELECTOR, 'h4.ametys-search-results__item-title a')
        for a in a_elements:
            hrefs.append(a.get_attribute('href')) if a.get_attribute('href') else None

        next_page = driver.find_elements(By.CSS_SELECTOR, 'a.ametys-pagination__arrow_next')
        
        if next_page:
            next_page[0].click()
            time.sleep(1)
        else:
            break  

    return hrefs


def scrape_title_formation(driver):
    """
    Retrieve the formation title (url already set in driver)
    """
    titles = driver.find_elements(By.CSS_SELECTOR, "h1.ametys-main-banner-alt__title")
    return [title.text for title in titles if title.text != ""][0]


def scrape_domain_formation(driver):
    """
    Retrieve the formation domain (url already set in driver)
    """
    domains = driver.find_elements(By.CSS_SELECTOR, "div.ametys-main-banner-alt__category")
    return [domain.text for domain in domains if domain.text != ""][0]


def scrape_table_formation(driver):
    """ 
    Retrieve the formation table informations (url already set in driver)
    """

    #table
    table = driver.find_element(By.CSS_SELECTOR, 'table.data')
    
    # tr
    rows = table.find_elements(By.TAG_NAME, 'tr')
    
    table_data = []
    for row in rows:
        row_data = []
        
        # th
        headers = row.find_elements(By.TAG_NAME, 'th')
        if headers:
            row_data += [header.text for header in headers]  # Ajouter les textes des <th>
        
        # td
        cells = row.find_elements(By.TAG_NAME, 'td')
        if cells:
            row_data += [cell.text for cell in cells if cell.text != ""]
        
        if row_data:
            table_data.append(row_data)

    #convert in dict
    dict_data = {}
    for data in table_data:
        dict_data[data[0]] = data[1]
    
    return dict_data


def scrape_formation(driver, url):
    """
    Return a dictionary where each key corresponds to a 
    specific information about the formation found at the URL
    """

    driver.get(url)
    time.sleep(1)

    dict = {}
    dict["title"] = scrape_title_formation(driver)
    dict["domain"] = scrape_domain_formation(driver)
    dict.update(scrape_table_formation(driver))
    return dict


def save_dict_to_json(data, filename="formations.json"):
    """
    Save a dictionary at JSON format
    """

    try:
        with open(filename, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        print(f"Success save {filename}")
    except Exception as e:
        print(f"Error save : {e}")
    
def main():
    all_formations_dict = {}

    # Init Selenium driver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    main_url = "https://formations.uha.fr/fr/par-diplomes/"
    for diploma_url in diplomas_urls:
        url = main_url + diploma_url + ".html"
        formations_links = scrape_formation_links(driver, url)
        
        for formation_link in formations_links:
            all_formations_dict[len(all_formations_dict)] = scrape_formation(driver, formation_link)
            
    driver.quit()
    
    save_dict_to_json(all_formations_dict)

if __name__ == "__main__":
    main()