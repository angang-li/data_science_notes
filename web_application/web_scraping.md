# Web scraping

<!-- TOC -->

- [Web scraping](#web-scraping)
    - [1. Beautiful soup scraping](#1-beautiful-soup-scraping)
        - [1.1. Web request with beautiful soup](#11-web-request-with-beautiful-soup)
        - [1.2. Extract html elements](#12-extract-html-elements)
        - [1.3. Loop through returned results](#13-loop-through-returned-results)
    - [2. Beautiful soup scraping with splinter](#2-beautiful-soup-scraping-with-splinter)
        - [2.1. Initiate browser](#21-initiate-browser)
        - [2.2. Scrape from browser with pagination](#22-scrape-from-browser-with-pagination)
    - [3. Pandas scraping](#3-pandas-scraping)
        - [3.1. Scrape HTML table to dataframe](#31-scrape-html-table-to-dataframe)
        - [3.2. From dataframe to HTML table](#32-from-dataframe-to-html-table)

<!-- /TOC -->

## 1. Beautiful soup scraping

### 1.1. Web request with beautiful soup

- Dependencies

    ```python
    from bs4 import BeautifulSoup
    import requests
    ```

- URL request

    ```python
    # URL of page to be scraped
    url = 'https://newjersey.craigslist.org/search/sss?sort=rel&query=guitar'

    # Retrieve page with the requests module
    response = requests.get(url)

    # HTML string
    html_string = response.text
    ```

- Create a Beautiful Soup object

    ```python
    soup = BeautifulSoup(html_string, 'html.parser')
    ```

- Print formatted version of the soup

    ```python
    print(soup.prettify())
    ```

### 1.2. Extract html elements

Beatiful soup is the best way to traverse the Document Object Model (DOM). DOM is how modern web browsers look at the HTML. The HTML is read in by the browser and converted to a more formalized data structure that helps the browser render the content.

- Extract an HTML element

    ```python
    # Extract the HTML title
    soup.title
    
    # Extract the text of the title
    soup.title.text
    ```

- Find first matching element

    ```python
    soup.body.p.text
    soup.body.find('p').text # equivalently
    ```

- Find all matching elements as iterable list

    ```python
    # Find by tag
    results = soup.body.find_all('p')
    results = soup.body.select('p') # equivalent
    ```

- Find matching elements with class or id selector

    ```python
    # Find by class
    soup.find_all(class_="result")
    soup.select(".result")

    # Find by class
    soup.find_all('li', class_="result")
    soup.select('li[class^="result"]')
    soup.select('.result')

    # Find by id
    soup.find_all('li', id="link2")
    soup.select('li#link2')
    soup.select('#link2')

    # Find by attribute
    soup.find_all('a', name="email")
    soup.find_all('a', attrs={"name": "email"}) # equivalent
    soup.select('a[name="email"]')
    ```

- Access the href attribute with bracket notation

    ```python
    thread = soup.find('li', class_='first')
    link = thread.a['href']
    ```

### 1.3. Loop through returned results

- Loop through returned results

    ```python
    for result in results:
        # Error handling
        try:
            # Identify and return title of listing
            title = result.find('a', class_="result-title").text
            # Identify and return price of listing
            price = result.a.span.text
            # Identify and return link to listing
            link = result.a['href']

            # Print results only if title, price, and link are available
            if (title and price and link):
                print('-------------')
                print(title)
                print(price)
                print(link)
        except Exception as e:
            print(e)
    ```

## 2. Beautiful soup scraping with splinter

`splinter`, built on top of `selenium`, simulates a real browser.

### 2.1. Initiate browser

- Dependencies
    ```python
    from splinter import Browser
    from bs4 import BeautifulSoup
    ```

- For Mac, find path of chromedriver
    ```python
    # https://splinter.readthedocs.io/en/latest/drivers/chrome.html
    !which chromedriver # /usr/local/bin/chromedriver
    ```

- Declare browser
    ```python
    executable_path = {'executable_path': '/usr/local/bin/chromedriver'}
    browser = Browser('chrome', **executable_path, headless=False)
    ```

### 2.2. Scrape from browser with pagination

- Browser visit URL

    ```python
    url = 'http://quotes.toscrape.com/'
    browser.visit(url)
    ```

- Scrape with pagination

    ```python
    for x in range(1, 6):

        html = browser.html
        soup = BeautifulSoup(html, 'html.parser')

        quotes = soup.find_all('span', class_='text')

        for quote in quotes:
            print('page:', x, '-------------')
            print(quote.text)

        browser.click_link_by_partial_text('Next')
    ```

- Quit browser

    ```python
    browser.quit()
    ```

## 3. Pandas scraping

### 3.1. Scrape HTML table to dataframe

Use `read_html` function in Pandas to automatically scrape any tabular data from a page

- Dependencies

    ```python
    import pandas as pd
    ```

- Read HTML tables

    ```python
    url = 'https://en.wikipedia.org/wiki/List_of_capitals_in_the_United_States'

    # Return a list of dataframes for any tabular data that Pandas found
    tables = pd.read_html(url)
    ```

- Format selected table

    ```python
    # slice off any of those dataframes that we want using normal indexing
    df = tables[0]
    df.columns = ['State', 'Abr.', 'State-hood Rank', 'Capital', 
                'Capital Since', 'Area (sq-mi)', 'Municipal Population', 'Metropolitan', 
                'Metropolitan Population', 'Population Rank', 'Notes']
    df.head()
    ```

### 3.2. From dataframe to HTML table

Use `to_html` method that we can use to generate HTML tables from DataFrames

- From DataFrames generate HTML table as string

    ```python
    html_table = df.to_html()
    html_table
    ```

- Strip unwanted newlines to clean up the table

    ```python
    html_table.replace('\n', '')
    ```

- Save the table directly to an HTML file

    ```python
    df.to_html('table.html')
    ```
