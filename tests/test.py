import requests
from bs4 import BeautifulSoup

url = "https://www.imobiliare.ro/vanzare-apartamente/bucuresti?pagina=1"
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "ro-RO,ro;q=0.9,en-US;q=0.8,en;q=0.7"
}

print("Łączę ze stroną...")
r = requests.get(url, headers=headers)
print(f"Status odpowiedzi HTTP: {r.status_code}")

if "Cloudflare" in r.text or "captcha" in r.text.lower() or "robot" in r.text.lower():
    print("🚨 STRONA NAS ZABLOKOWAŁA (Ściana Cloudflare / Captcha)!")
else:
    print("✅ Strona załadowana pomyślnie. Szukam linków...")
    soup = BeautifulSoup(r.text, "html.parser")
    links = [a['href'] for a in soup.find_all('a', href=True) if 'oferta' in a['href'].lower()]

    print(f"Znaleziono {len(links)} potencjalnych linków do ogłoszeń. Oto pierwsze 3:")
    for link in links[:3]:
        print(link)