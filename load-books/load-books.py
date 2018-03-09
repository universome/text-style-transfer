import sys
import json
import pickle
import time

from pyquery import PyQuery as pq
import requests
from tqdm import tqdm


API_URL = 'http://api.knigafund.ru/api'
API_KEY = 'api-example'
AUTHORS_SEARCH_URL = '{}/authors/search.json'.format(API_URL)
AUTHOR_BOOKS_URL = '{}/authors/{{}}/books.json'.format(API_URL)


def main(cmd):
    if cmd == 'load_authors_ids':
        load_authors_ids()
    elif cmd == 'load_books_lists':
        load_books_lists()
    elif cmd == 'load_books_page_urls':
        load_books_page_urls()
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


if __name__ == '__main__':
    cmd = sys.argv[1]
    print('Command to run:', cmd)
    main(cmd)
