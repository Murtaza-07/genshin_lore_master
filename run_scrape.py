import asyncio
from pathlib import Path
from main import MultiSourceScraper, EnhancedVectorDB

data_dir = Path("genshin_data")
scraper = MultiSourceScraper(data_dir)

print("Starting manual wiki scraping...")
wiki_entries = asyncio.run(scraper.scrape_wiki_async())
print(f"Manual wiki scraping completed. Found {len(wiki_entries)} wiki entries.")

updated_entry_ids = []
if wiki_entries:
    new_wiki_entries, new_wiki_chunks = scraper.save_entries_with_chunks(wiki_entries)
    if new_wiki_entries > 0 or new_wiki_chunks > 0:
        updated_entry_ids.extend([e.entry_id for e in wiki_entries])
        print(f"Wiki: Added/Updated {new_wiki_entries} entries, {new_wiki_chunks} chunks.")

vector_db = EnhancedVectorDB(Path("genshin_data"))
if updated_entry_ids:
    vector_db.rebuild_index_from_db(scraper.db_path)
    print("Vector database index rebuilt due to new/updated content.")
else:
    print("No new content, vector database index not rebuilt.")

print("Manual wiki scraping and update process finished.")