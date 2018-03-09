import json
import requests


API_URL = 'http://api.knigafund.ru/api'
API_KEY = 'api-example'
AUTHORS_SEARCH_QUERY = '{}/authors/search.json'.format(API_URL)

def main():
    authors = open('load-books/authors.txt', encoding='utf-8').read().splitlines()
    author_to_id = {}

    # Obtaining authors ids
    for author in authors:
        # We search with only second name, because their search engine is dumb
        search_params = {'api-key': API_KEY, 'query': author.split()[0]}
        response = requests.get(url=AUTHORS_SEARCH_QUERY, params=search_params)
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

    with open('load-books/authors-ids.txt', 'w', encoding='utf-8') as f:
        for author in author_to_id:
            f.write(author + '|' + str(author_to_id[author]) + '\n')


if __name__ == '__main__':
    main()
