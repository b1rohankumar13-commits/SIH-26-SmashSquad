"""Collect official seasonal source documents for the approved event categories."""
import acquire_imd_confirmed as core
from bs4 import BeautifulSoup
from urllib.parse import urljoin

core.log = core.root / 'data/metadata/imd_event_reports_2021_2023.jsonl'
articles = [
    (2021,'winter','https://mausamjournal.imd.gov.in/index.php/MAUSAM/article/view/5094'),
    (2021,'hot_weather','https://mausamjournal.imd.gov.in/index.php/MAUSAM/article/view/5911'),
    (2021,'monsoon','https://mausamjournal.imd.gov.in/index.php/MAUSAM/article/view/5954'),
    (2021,'post_monsoon','https://mausamjournal.imd.gov.in/index.php/MAUSAM/article/view/6041'),
    (2022,'winter','https://mausamjournal.imd.gov.in/index.php/MAUSAM/article/view/6131'),
    (2022,'hot_weather','https://mausamjournal.imd.gov.in/index.php/MAUSAM/article/view/6225'),
    (2022,'monsoon','https://mausamjournal.imd.gov.in/index.php/MAUSAM/article/view/6313'),
    (2022,'post_monsoon','https://mausamjournal.imd.gov.in/index.php/MAUSAM/article/view/6405'),
    (2023,'winter','https://mausamjournal.imd.gov.in/index.php/MAUSAM/article/view/6516'),
    (2023,'hot_weather','https://mausamjournal.imd.gov.in/index.php/MAUSAM/article/view/6602'),
    (2023,'monsoon','https://mausamjournal.imd.gov.in/index.php/MAUSAM/article/view/6723'),
    (2023,'post_monsoon','https://mausamjournal.imd.gov.in/index.php/MAUSAM/article/view/6848'),
    (2024,'winter','https://mausamjournal.imd.gov.in/index.php/MAUSAM/article/view/6947'),
]

def fetch_pdf(page):
    result = core.request('GET',page)
    soup = BeautifulSoup(result.text,'html.parser')
    meta = soup.find('meta',attrs={'name':'citation_pdf_url'})
    if meta and meta.get('content'):
        result = core.request('GET',urljoin(result.url,meta['content']))
    else:
        link = next(a for a in soup.find_all('a',href=True) if a.get_text(strip=True).lower() in ('pdf','download pdf','download download pdf'))
        result = core.request('GET',urljoin(result.url,link['href']))
    if not result.content.startswith(b'%PDF-'):
        soup = BeautifulSoup(result.text,'html.parser')
        link = next(a for a in soup.find_all('a',href=True) if '/article/download/' in a['href'])
        result = core.request('GET',urljoin(result.url,link['href']))
    return result

for year,season,page in articles:
    core.acquire(core.raw/'official_events'/str(year)/f'weather_in_india_{season}_{year}.pdf','pdf',year,page,lambda p=page:fetch_pdf(p))

page = 'https://www.imdpune.gov.in/Reports/monsoon_reports.html'
try:
    response = core.request('GET',page)
    soup = BeautifulSoup(response.text,'html.parser')
    for year in (2021,2022,2023):
        matches = []
        for tag in soup.find_all(['a','option']):
            value = tag.get('href') or tag.get('value') or ''
            if str(year) in tag.get_text() + value and '.pdf' in value.lower():
                matches.append(urljoin(response.url,value))
        if len(set(matches)) != 1:
            core.record({'product':'monsoon_report','year':year,'status':'discovery_unresolved','source_page':page})
            continue
        url = matches[0]
        core.acquire(core.raw/'official_events'/str(year)/f'monsoon_report_{year}.pdf','pdf',year,page,lambda u=url:core.request('GET',u))
except Exception as e:
    core.record({'product':'monsoon_report_discovery','status':'failed','error':str(e)})
print('Official event-report acquisition pass finished.',flush=True)
