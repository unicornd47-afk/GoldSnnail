import { requestUrl, Notice } from 'obsidian';

export class CodeCrawler {
    // Verhindert Endlosschleifen bei zirkulären Abhängigkeiten
    private visitedUrls: Set<string> = new Set();
    
    // Speichert unsere extrahierten Rohtexte (URL -> Inhalt)
    public crawledData: Map<string, string> = new Map();

    // Nur diese Dateitypen interessieren uns
    private targetExtensions = ['.js', '.json', '.rs', '.html'];

    /**
     * Startet den Crawl-Prozess ab einer Start-URL.
     */
    public async crawl(startUrl: string, maxDepth: number = 2): Promise<void> {
        new Notice(`Starte Crawling für: ${startUrl}`);
        await this.processUrl(startUrl, 0, maxDepth);
        new Notice(`Crawling beendet. ${this.crawledData.size} Dateien gefunden.`);
    }

    /**
     * Rekursive Funktion zum Abrufen und Weiterverarbeiten.
     */
    private async processUrl(url: string, currentDepth: number, maxDepth: number): Promise<void> {
        // Abbruchbedingungen: Tiefe erreicht oder URL schon besucht
        if (currentDepth > maxDepth || this.visitedUrls.has(url)) {
            return;
        }

        this.visitedUrls.add(url);

        try {
            console.log(`Fetching: ${url}`);
            
            // Obsidians interne API umgeht CORS
            const response = await requestUrl({ url: url });
            
            if (response.status !== 200) {
                console.warn(`Fehler ${response.status} bei ${url}`);
                return;
            }

            const content = response.text;
            this.crawledData.set(url, content);

            // Wenn wir das Limit noch nicht erreicht haben, suchen wir nach weiteren Links
            if (currentDepth < maxDepth) {
                const newUrls = this.extractUrlsFromContent(url, content);
                
                // Paralleles Crawlen der gefundenen Links
                const crawlPromises = newUrls.map(nextUrl => 
                    this.processUrl(nextUrl, currentDepth + 1, maxDepth)
                );
                await Promise.all(crawlPromises);
            }

        } catch (error) {
            console.error(`Fehler beim Crawlen von ${url}:`, error);
        }
    }

    /**
     * Hilfsfunktion: Sucht rudimentär nach neuen URLs im Text.
     * (Wird später durch unsere echten AST-Parser ersetzt!)
     */
    private extractUrlsFromContent(baseUrl: string, content: string): string[] {
        const foundUrls: string[] = [];
        
        // Sehr simpler Regex für href="...", src="..." oder import "...";
        const urlRegex = /(?:href|src|import\s+['"]|from\s+['"])=?['"]([^'"]+)['"]/g;
        let match;

        while ((match = urlRegex.exec(content)) !== null) {
            let extractedPath = match[1];
            
            try {
                // Relativen Pfad in absolute URL umwandeln
                const absoluteUrl = new URL(extractedPath, baseUrl).href;
                
                // Prüfen, ob die Endung passt (oder es eine HTML/Root-Seite ist)
                const urlObj = new URL(absoluteUrl);
                const hasValidExtension = this.targetExtensions.some(ext => urlObj.pathname.endsWith(ext));
                const isLikelyHtml = !urlObj.pathname.includes('.'); // Z.B. /about
                
                if (hasValidExtension || isLikelyHtml) {
                    foundUrls.push(absoluteUrl);
                }
            } catch (e) {
                // Ignoriere ungültige URLs wie "mailto:" oder "#section"
            }
        }

        return foundUrls;
    }
}
