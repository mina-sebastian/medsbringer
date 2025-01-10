import scrapy
from medsbringer.items import MedsbringerItem
from scrapy.http import Response
import pandas as pd

class AnmSpider(scrapy.Spider):
    name = 'anm'
    allowed_domains = ['anm.ro']
    start_urls = ['https://nomenclator.anm.ro/medicamente?page=' + str(x) for x in range(1541, 1542)]

    # List to hold items
    items_list = []

    def parse(self, response: Response):
        rows = response.css('table.table tbody tr')
        for row in rows:
            item = MedsbringerItem()
            item['nume'] = row.css('td:nth-child(2)::text').get()
            item['firma'] = row.css('td:nth-child(7)::text').get()
            
            pro_link = row.css('button::attr(data-linkpro)').get()
            item['url'] = pro_link
            self.items_list.append(dict(item))  # Store item in the list
            yield item

    def close(self, spider):
        # Save items to an Excel file
        print(f'Items scraped: {len(self.items_list)}')
        # df = pd.DataFrame(self.items_list)
        # df.to_excel('medsbringer_items.xlsx', index=False, sheet_name='Medications')
