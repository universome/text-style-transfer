import os
import sys
import json
import pickle
import time
import zipfile
from urllib.request import urlretrieve

from pyquery import PyQuery as pq
import requests
from tqdm import tqdm


API_URL = 'http://api.knigafund.ru/api'
API_KEY = 'api-example'
AUTHORS_SEARCH_URL = '{}/authors/search.json'.format(API_URL)
AUTHOR_BOOKS_URL = '{}/authors/{{}}/books.json'.format(API_URL)
BOOK_SEARCH_QUERY = 'https://aldebaran.ru/pages/biblio_search/?q={}'
BOOKS_HOST_URL = 'https://aldebaran.ru'
HEADERS = {
    'Host': 'aldebaran.ru',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9,ru;q=0.8',
}


def main(cmd):
    if cmd == 'load_authors_ids':
        load_authors_ids()
    elif cmd == 'load_books_lists':
        load_books_lists()
    elif cmd == 'load_books_page_urls':
        load_books_page_urls()
    elif cmd == 'download_books':
        download_books()
    elif cmd == 'unzip_books':
        unzip_books()
    elif cmd == 'convert_books_to_txt':
        convert_books_to_txt()
    elif cmd == 'all':
        load_authors_ids()
        load_books_lists()
        load_books_page_urls()
        download_books()
        unzip_books()
        convert_books_to_txt()
    else:
        raise NotImplemented


def load_authors_ids():
    authors = open('load-books/authors.txt', encoding='utf-8').read().splitlines()
    author_to_id = {}

    # Obtaining authors ids
    for author in authors:
        # We search with only second name, because their search engine is dumb
        search_params = {'api-key': API_KEY, 'query': author.split()[0]}
        response = requests.get(url=AUTHORS_SEARCH_URL, params=search_params)
        result = response.json()

        if result['info']['count'] == 0:
            print('Not found anything for', author)
            continue

        if result['info']['count'] == 1:
            author_to_id[author] = result['authors'][0]['author']['id']
            continue

        # Ok, let's guess the author
        initials = author.split()[0] + ' ' + author.split()[1][0] + '.'

        for author_info in result['authors']:
            author_name = author_info['author']['name']

            if initials in author_name:
                print('Found occurence:', initials, '->', author_name)
                author_to_id[author] = author_info['author']['id']
                break

    # FIXME
    author_to_id['Толстой Лев'] = 4539

    with open('load-books/author-to-id.json', 'w', encoding='utf-8') as f:
        json.dump(author_to_id, f)


def load_books_lists():
    author_to_id = json.load(open('load-books/author-to-id.json', encoding='utf-8'))
    author_to_books = {}

    for author in author_to_id:
        books = []
        search_params = {'api-key': API_KEY, 'per_page': 100}
        url = AUTHOR_BOOKS_URL.format(author_to_id[author])
        response = requests.get(url=url, params=search_params)
        result = response.json()

        for book_info in result['books']:
            book = book_info['book']

            if not 'pages_count' in book: continue
            if book['pages_count'] < 50: continue

            books.append(book['name'])

        author_to_books[author] = list(set(books))

        # Let's save results after each author just in case we are banned
        with open('load-books/author-to-books.json', 'w', encoding='utf-8') as f:
            json.dump(author_to_books, f)


def load_books_page_urls():
    author_to_books = json.load(open('load-books/author-to-books.json', encoding='utf-8'))
    titles = [a + ' ' + b for a in author_to_books for b in author_to_books[a]]
    with open('load-books/books-page-urls.json', 'r', encoding='utf-8') as f: books_urls = json.load(f)
    print('Num books to download', len(titles))
    print('Already have books', len(books_urls))

    for title in tqdm(titles):
        time.sleep(5) # Do not DDOS
        url = BOOK_SEARCH_QUERY.format(title)
        doc = pq(url=url, headers=HEADERS)

        if 'робот' in doc.text(): print('We are banned :(')
        if len(doc('.item_info.border_bottom')) < 2: continue
        if len(doc('.item_info.border_bottom:nth-child(2) a')) != 2: continue

        href = doc('.item_info.border_bottom:nth-child(2) a')[0].attrib['href']
        if not ((BOOKS_HOST_URL + href) in books_urls):
            books_urls.append(BOOKS_HOST_URL + href)
        else:
            print('Already have', BOOKS_HOST_URL + href)

        # Let's save after each book in case of we are banned
        with open('load-books/books-page-urls.json', 'w', encoding='utf-8') as f:
            json.dump(books_urls, f)


# def load_books_file_urls():
#     books_urls = json.load(open('load-books/books-page-urls.json', encoding='utf-8'))
#     print('Num books to retrieve:', len(books_urls))

#     for url in tqdm(books_urls):
#         time.sleep(5)
#         doc = pq(url=url, headers=HEADERS)

#         if len(doc('.wrapper .block_mixed .bm1 .item_info.border_bottom')) != 1: continue
#         if len(doc('.wrapper .block_mixed .bm1 .item_info.border_bottom a:nth-child(2)')) != 1: continue

#         el = doc('.wrapper .block_mixed .bm1 .item_info.border_bottom a:nth-child(2)')
#         if el.text() != 'html.zip':
#             print('Found unusual format:', el.text(), 'for url:', url)
#             continue

#         books_file_urs.append(BOOKS_HOST_URL + el.attrib['href'])

#         with open('load-books/books-file-urls.json', 'w', encoding='utf-8') as f:
#             json.dump(books_file_urs, f)


def download_books():
    books_urls = json.load(open('load-books/books-page-urls.json', encoding='utf-8'))
    books_urls = [l for l in books_urls if not ('litres' in l or 'tags' in l)]
    print('Num books to download:', len(books_urls))
    if not os.path.exists('load-books/archives'): os.mkdir('load-books/archives')

    for page_url in tqdm(books_urls):
        time.sleep(15)
        download_url = page_url + 'download.html.zip'
        print('Downloading', download_url)

        try:
            file_name = 'load-books/archives/{}_{}.zip'.format(*page_url.split('/')[-3:-1])
            urlretrieve(download_url, filename=file_name)
        except Exception as e:
            print('Could not download book:', download_url)
            print(e)


def unzip_books():
    if not os.path.exists('load-books/html'): os.mkdir('load-books/html')
    archives = [a for a in os.listdir('load-books/archives') if b[-4:] == '.zip']

    for archive in archives:
        try:
            archive_path = 'load-books/archives/' + archive
            zip_ref = zipfile.ZipFile(archive_path, 'r')
            zip_ref.extractall('load-books/html')
            zip_ref.close()
        except Exception as e:
            print('Could not extract book:', archive_path)
            print(e)

    # Let's clean directory from images
    for file in os.listdir('load-books/html'):
        if file[-5:] != '.html':
            os.remove('load-books/html/' + file)


def convert_books_to_txt():
    if not os.path.exists('load-books/txt'): os.mkdir('load-books/txt')
    book_names = [b[:-5] for b in os.listdir('load-books/html') if b[-5:] == '.html']

    for book_name in book_names:
        in_book_path = 'load-books/html/' + book_name + '.html'
        out_book_path = 'load-books/txt/' + book_name + '.txt'

        with open(in_book_path, 'rb') as fin, open(out_book_path, 'w', encoding='utf-8') as fout:
            try:
                doc = pq(fin.read())
                doc('style').remove()
                doc('title').remove()
                fout.write(doc.text())
            except Exception as e:
                print('Could not parse book:', in_book_path)
                print(e)


if __name__ == '__main__':
    cmd = sys.argv[1]
    print('Command to run:', cmd)
    main(cmd)
