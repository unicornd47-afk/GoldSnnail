import { Plugin } from 'obsidian';
import { CodeCrawler } from './Crawler';

export default class ArchitectureMapperPlugin extends Plugin {
    async onload() {
        this.addCommand({
            id: 'start-code-crawler',
            name: 'Crawl Website/Repo for Architecture',
            callback: async () => {
                const crawler = new CodeCrawler();
                // Hier könntest du später einen Modal-Dialog einbauen, 
                // der den User nach der URL fragt.
                await crawler.crawl('https://example.com/some/repo/index.html', 2);
                
                console.log("Ergebnisse:", crawler.crawledData);
            }
        });
    }
}
