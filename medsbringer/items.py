# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class MedsbringerItem(scrapy.Item):
    nume = scrapy.Field()
    url = scrapy.Field()
    firma = scrapy.Field()