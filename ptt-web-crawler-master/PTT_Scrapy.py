from PttWebCrawler.crawler import *

c = PttWebCrawler(as_lib=True)
c.parse_articles(6001, 6799, 'Stock')

#2410為2021年初6799為2023年底
#