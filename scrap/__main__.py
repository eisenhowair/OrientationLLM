from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

diplomes_urls = ["b-u-t", 
            "licences", 
            "licences-professionnelles", 
            "masters", 
            "antenne-de-l-ecole-doctorale",
            "diplomes-d-ingenieurs-cycles-preparatoires",
            "diplomes-d-universite"]

"""
retrieve all links formation
"""
def scrape_formation_links(driver, url):
    driver.get(url)
    time.sleep(1)
    
    titles = driver.find_elements(By.CSS_SELECTOR, "h4.ametys-search-results__item-title a")
    hrefs = [title.get_attribute('href') for title in titles if title.get_attribute('href')]
    
    return hrefs

"""
retrieve title formation (url already set in driver)
"""
def scrape_title_formation(driver):
    titles = driver.find_elements(By.CSS_SELECTOR, "h1.ametys-main-banner-alt__title")
    return [title.text for title in titles if title.text != ""][0]

# retrieve domain formation (url already set in driver)
def scrape_domain_formation(driver):
    domains = driver.find_elements(By.CSS_SELECTOR, "div.ametys-main-banner-alt__category")
    return [domain.text for domain in domains if domain.text != ""][0]

""" 
retrieve table informations formation (url already set in driver)
"""
def scrape_table_formation(driver):
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

"""
return a dictionary where each key corresponds to a 
specific information about the formation found at the URL
"""
def scrape_formation(driver, url):
    driver.get(url)
    time.sleep(1)

    dict = {}
    dict["title"] = scrape_title_formation(driver)
    dict["domain"] = scrape_domain_formation(driver)
    dict.update(scrape_table_formation(driver))
    return dict
    
def main():
    all_formations_dict = {}

    # Initialisation du driver Selenium
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    main_url = "https://formations.uha.fr/fr/par-diplomes/"
    for diplome_url in diplomes_urls:
        url = main_url + diplome_url + ".html"
        formations_links = scrape_formation_links(driver, url)
        
        for formation_link in formations_links:
            all_formations_dict[len(all_formations_dict)] = scrape_formation(driver, formation_link)
            
    driver.quit()
    print(all_formations_dict)
    return all_formations_dict

