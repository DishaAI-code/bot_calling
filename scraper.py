import requests
from bs4 import BeautifulSoup

def scrape_lpu_courses():
    url = "https://www.lpu.in/landing-pages/brand/"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Updated selector - look for course listings
        courses = []
        course_elements = soup.select('.program-list li') or soup.select('.course-list li')
        
        for element in course_elements:
            courses.append(element.get_text(strip=True))
            
        if not courses:
            print("Warning: No courses found using default selectors. Trying alternatives...")
            # Alternative selectors if primary fails
            course_containers = soup.find_all('div', class_='course-item')
            for container in course_containers:
                title = container.find('h3')
                if title:
                    courses.append(title.get_text(strip=True))
        
        return courses
    
    except Exception as e:
        print(f"Error scraping LPU courses: {e}")
        return []

if __name__ == "__main__":
    courses = scrape_lpu_courses()
    print("Courses Offered:")
    for i, course in enumerate(courses, 1):
        print(f"{i}. {course}")