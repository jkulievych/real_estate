import undetected_chromedriver as uc
import time
from bs4 import BeautifulSoup

print("1. Uruchamiam niewykrywalną przeglądarkę Chrome...")
# Inicjalizacja przeglądarki
driver = uc.Chrome()

url = "https://www.imobiliare.ro/vanzare-apartamente/bucuresti?pagina=1"
print(f"2. Wchodzę na stronę: {url}")
driver.get(url)

print("3. Czekam 10 sekund, żeby Cloudflare nas przepuścił...")
time.sleep(10)

# Pobieramy kod strony po tym, jak przeglądarka załadowała wszystko
html = driver.page_source
soup = BeautifulSoup(html, "html.parser")

# Szukamy linków
links = []
for a in soup.find_all("a", href=True):
    href = a["href"]
    if "/oferta/" in href and "-" in href:
        links.append(href)

# Pozbywamy się duplikatów i liczymy
unique_links = set(links)

if len(unique_links) > 0:
    print(f"\n✅ SUKCES! Przebiliśmy się przez zaporę. Znaleziono {len(unique_links)} unikalnych linków do ogłoszeń na tej stronie.")
else:
    print("\n❌ Niestety, nadal blokada. Zobaczyłaś okienko z Captchą do kliknięcia?")

print("\nZamykam przeglądarkę za 3 sekundy...")
time.sleep(3)
driver.quit()