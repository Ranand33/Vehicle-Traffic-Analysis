import os
import requests
import time
import csv
import argparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from concurrent.futures import ThreadPoolExecutor
import urllib.parse
import random

class VehicleImageCollector:
    def __init__(self, output_dir, makes_models_file=None, num_images=100, headless=True):
        """
        Initialize the Vehicle Image Collector
        
        Args:
            output_dir (str): Directory to save collected images
            makes_models_file (str): CSV file with makes and models to collect
            num_images (int): Number of images to collect per make/model
            headless (bool): Run browser in headless mode
        """
        self.output_dir = output_dir
        self.makes_models_file = makes_models_file
        self.num_images = num_images
        self.headless = headless
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize browser for searches that require JavaScript
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36")
        
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        
        # URLs for image search
        self.search_engines = {
            "google": "https://www.google.com/search?q={}&tbm=isch",
            "bing": "https://www.bing.com/images/search?q={}",
        }
        
        # Default makes and models if no file is provided
        self.default_makes_models = [
            ("Toyota", ["Camry", "Corolla", "RAV4", "Prius"]),
            ("Honda", ["Civic", "Accord", "CR-V", "Pilot"]),
            ("Ford", ["F-150", "Mustang", "Explorer", "Escape"]),
            ("Chevrolet", ["Silverado", "Malibu", "Equinox", "Tahoe"]),
            ("Nissan", ["Altima", "Rogue", "Sentra", "Pathfinder"]),
            ("Hyundai", ["Elantra", "Sonata", "Tucson", "Santa Fe"]),
            ("BMW", ["3 Series", "5 Series", "X3", "X5"]),
            ("Mercedes-Benz", ["C-Class", "E-Class", "GLC", "GLE"]),
            ("Audi", ["A4", "A6", "Q5", "Q7"]),
            ("Tesla", ["Model 3", "Model Y", "Model S", "Model X"])
        ]
    
    def load_makes_models(self):
        """
        Load makes and models from CSV file or use defaults
        
        Returns:
            list: List of (make, models) tuples
        """
        if self.makes_models_file and os.path.exists(self.makes_models_file):
            makes_models = []
            with open(self.makes_models_file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header
                
                current_make = None
                current_models = []
                
                for row in reader:
                    if len(row) >= 2:
                        make, model = row[0].strip(), row[1].strip()
                        
                        if make and model:
                            if current_make != make and current_make is not None:
                                makes_models.append((current_make, current_models))
                                current_models = []
                            
                            current_make = make
                            current_models.append(model)
                
                if current_make and current_models:
                    makes_models.append((current_make, current_models))
            
            return makes_models
        else:
            return self.default_makes_models
    
    def search_images(self, make, model, engine="google"):
        """
        Search for images of a specific vehicle make and model
        
        Args:
            make (str): Vehicle make
            model (str): Vehicle model
            engine (str): Search engine to use ("google" or "bing")
            
        Returns:
            list: List of image URLs
        """
        query = f"{make} {model} car exterior"
        encoded_query = urllib.parse.quote(query)
        search_url = self.search_engines[engine].format(encoded_query)
        
        # Open the search page
        self.driver.get(search_url)
        
        # Wait for images to load
        time.sleep(2)
        
        # Scroll down to load more images
        for _ in range(5):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
        
        # Extract image URLs based on search engine
        image_urls = []
        
        if engine == "google":
            # For Google Images
            elements = self.driver.find_elements(By.CSS_SELECTOR, "img.rg_i")
            for element in elements:
                try:
                    element.click()
                    time.sleep(0.5)
                    
                    # Look for the larger image URL
                    large_images = self.driver.find_elements(By.CSS_SELECTOR, "img.n3VNCb")
                    for img in large_images:
                        src = img.get_attribute("src")
                        if src and src.startswith("http") and src not in image_urls:
                            image_urls.append(src)
                except:
                    pass
        
        elif engine == "bing":
            # For Bing Images
            elements = self.driver.find_elements(By.CSS_SELECTOR, ".mimg")
            for element in elements:
                try:
                    src = element.get_attribute("src")
                    if src and src.startswith("http") and src not in image_urls:
                        image_urls.append(src)
                except:
                    pass
        
        print(f"Found {len(image_urls)} images for {make} {model}")
        return image_urls[:self.num_images]  # Limit to requested number
    
    def download_image(self, url, save_path):
        """
        Download an image from a URL
        
        Args:
            url (str): Image URL
            save_path (str): Path to save the image
            
        Returns:
            bool: True if download was successful, False otherwise
        """
        try:
            # Add random delay to avoid being blocked
            time.sleep(random.uniform(0.5, 1.5))
            
            # Set up headers to mimic browser request
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                return False
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False
    
    def collect_images(self):
        """
        Collect vehicle images for all makes and models
        """
        makes_models = self.load_makes_models()
        total_makes_models = sum(len(models) for _, models in makes_models)
        
        print(f"Starting image collection for {total_makes_models} make/model combinations")
        
        for make, models in makes_models:
            make_dir = os.path.join(self.output_dir, make)
            os.makedirs(make_dir, exist_ok=True)
            
            for model in models:
                # Create directory for this model
                model_dir = os.path.join(make_dir, model)
                os.makedirs(model_dir, exist_ok=True)
                
                print(f"Collecting images for {make} {model}")
                
                # Search for images
                image_urls = []
                for engine in self.search_engines.keys():
                    urls = self.search_images(make, model, engine)
                    image_urls.extend(urls)
                    
                    # Break if we have enough images
                    if len(image_urls) >= self.num_images:
                        break
                
                # Deduplicate URLs
                image_urls = list(set(image_urls))[:self.num_images]
                
                # Download images in parallel
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = []
                    
                    for i, url in enumerate(image_urls):
                        save_path = os.path.join(model_dir, f"{i:03d}.jpg")
                        future = executor.submit(self.download_image, url, save_path)
                        futures.append((future, url, save_path))
                    
                    # Process results
                    successful = 0
                    for future, url, path in futures:
                        if future.result():
                            successful += 1
                
                print(f"Downloaded {successful} images for {make} {model}")
    
    def close(self):
        """
        Close the browser
        """
        self.driver.quit()

def main():
    parser = argparse.ArgumentParser(description='Collect vehicle images for training a make/model recognition model')
    parser.add_argument('--output', type=str, default='/home/plato/Documents/Projects/Car Detection/data', help='Output directory')
    parser.add_argument('--csv', type=str, default=None, help='CSV file with makes and models')
    parser.add_argument('--num_images', type=int, default=100, help='Number of images per make/model')
    parser.add_argument('--no_headless', action='store_true', help='Disable headless browser mode')
    
    args = parser.parse_args()
    
    collector = VehicleImageCollector(
        output_dir=args.output,
        makes_models_file=args.csv,
        num_images=args.num_images,
        headless=not args.no_headless
    )
    
    try:
        collector.collect_images()
    finally:
        collector.close()
    
    print("Image collection complete")

if __name__ == "__main__":
    main()