import requests
from bs4 import BeautifulSoup

url_string = "https://www.aclweb.org/anthology/P19-"

# write to this md file

f = open("abstracts.md", 'a+')


# implement the loop here
for i in range (1524, 1660):
    url = url_string + str(i) + '/'
    print(url)
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html5lib')
    
    title = soup.find('h2', attrs={'id':'title'})
    f.write('## ' + title.text.strip() + '\n')

    f.write('- ' + url + '\n\n')
    
    f.write('```\n')
    abstract = soup.find('div', attrs={'class':'card-body acl-abstract'})
    f.write(abstract.text.strip() + '\n')
    f.write('```\n\n')

