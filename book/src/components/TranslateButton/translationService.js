/**
 * Translation Service for Docusaurus Book
 *
 * This service handles content translation between languages,
 * with a focus on technical content related to Physical AI and Humanoid Robotics.
 */

class TranslationService {
  constructor() {
    this.apiKey = process.env.REACT_APP_TRANSLATION_API_KEY || null;
    this.baseURL = process.env.REACT_APP_TRANSLATION_API_URL || 'https://api-free.deepl.com/v2';
    this.isDeepL = true; // Default to DeepL API, can be configured for other services
  }

  /**
   * Translate text using the configured translation service
   * @param {string} text - Text to translate
   * @param {string} targetLang - Target language code (e.g., 'ur', 'es', 'fr')
   * @param {string} sourceLang - Source language code (default: 'en')
   * @returns {Promise<string>} - Translated text
   */
  async translateText(text, targetLang, sourceLang = 'en') {
    if (!text || !targetLang) {
      throw new Error('Text and target language are required');
    }

    // For Urdu specifically, ensure proper handling
    const targetCode = this.normalizeLanguageCode(targetLang);

    try {
      if (this.isDeepL) {
        return await this.translateWithDeepL(text, targetCode, sourceLang);
      } else {
        // Fallback to a different service if needed
        return await this.translateWithAlternativeService(text, targetCode, sourceLang);
      }
    } catch (error) {
      console.error('Translation error:', error);
      // Return original text if translation fails
      return text;
    }
  }

  /**
   * Translate text using DeepL API
   * @param {string} text - Text to translate
   * @param {string} targetLang - Target language code
   * @param {string} sourceLang - Source language code
   * @returns {Promise<string>} - Translated text
   */
  async translateWithDeepL(text, targetLang, sourceLang) {
    if (!this.apiKey) {
      throw new Error('DeepL API key is required for translation');
    }

    // DeepL specific language code mapping
    const deepLTargetLang = this.mapToDeepLCode(targetLang);
    const deepLSourceLang = this.mapToDeepLCode(sourceLang);

    const response = await fetch(`${this.baseURL}/translate`, {
      method: 'POST',
      headers: {
        'Authorization': `DeepL-Auth-Key ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: [text],
        target_lang: deepLTargetLang,
        source_lang: deepLSourceLang,
      }),
    });

    if (!response.ok) {
      throw new Error(`DeepL API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    if (data.translations && data.translations.length > 0) {
      return data.translations[0].text;
    } else {
      throw new Error('No translation returned from DeepL API');
    }
  }

  /**
   * Translate with an alternative service (placeholder for Google Translate API)
   * @param {string} text - Text to translate
   * @param {string} targetLang - Target language code
   * @param {string} sourceLang - Source language code
   * @returns {Promise<string>} - Translated text
   */
  async translateWithAlternativeService(text, targetLang, sourceLang) {
    // This is a placeholder - in a real implementation you would use
    // Google Translate API, AWS Translate, or another service
    console.warn('Using alternative translation service - implement actual API call');

    // For now, return original text with a note
    return text;
  }

  /**
   * Translate page content elements
   * @param {string} targetLang - Target language code
   * @returns {Promise<void>}
   */
  async translatePageContent(targetLang) {
    // Get all translatable elements on the page
    const elements = document.querySelectorAll('[data-translatable="true"], h1, h2, h3, h4, h5, h6, p, li, td, th, .markdown p, .markdown li, .markdown td, .markdown th');

    // Process each element
    for (const element of elements) {
      const originalText = element.getAttribute('data-original-text') || element.textContent;

      // Skip if element is already translated or empty
      if (!originalText.trim() || element.classList.contains('translated')) {
        continue;
      }

      // Store original text for future translations
      element.setAttribute('data-original-text', originalText);

      // Translate the text
      const translatedText = await this.translateText(originalText, targetLang);

      // Update the element with translated text
      element.textContent = translatedText;
      element.classList.add('translated');
      element.setAttribute('data-translated-lang', targetLang);
    }
  }

  /**
   * Translate specific content blocks
   * @param {Array<string>} texts - Array of texts to translate
   * @param {string} targetLang - Target language code
   * @returns {Promise<Array<string>>} - Array of translated texts
   */
  async translateMultipleTexts(texts, targetLang) {
    const translations = [];

    for (const text of texts) {
      try {
        const translated = await this.translateText(text, targetLang);
        translations.push(translated);
      } catch (error) {
        console.error('Error translating text:', error);
        translations.push(text); // Fallback to original text
      }
    }

    return translations;
  }

  /**
   * Normalize language codes to standard format
   * @param {string} lang - Language code
   * @returns {string} - Normalized language code
   */
  normalizeLanguageCode(lang) {
    const mapping = {
      'urdu': 'ur',
      'english': 'en',
      'spanish': 'es',
      'french': 'fr',
      'german': 'de',
      'chinese': 'zh',
      'ur': 'ur',
      'en': 'en',
      'es': 'es',
      'fr': 'fr',
      'de': 'de',
      'zh': 'zh'
    };

    return mapping[lang.toLowerCase()] || lang.toLowerCase();
  }

  /**
   * Map language codes to DeepL format
   * @param {string} lang - Language code
   * @returns {string} - DeepL compatible language code
   */
  mapToDeepLCode(lang) {
    const mapping = {
      'ur': 'UR',
      'en': 'EN',
      'es': 'ES',
      'fr': 'FR',
      'de': 'DE',
      'zh': 'ZH'
    };

    return mapping[lang.toLowerCase()] || lang.toUpperCase();
  }

  /**
   * Get supported languages
   * @returns {Array} - Array of supported language objects
   */
  getSupportedLanguages() {
    return [
      { code: 'en', name: 'English' },
      { code: 'ur', name: 'Urdu' },
      { code: 'es', name: 'Spanish' },
      { code: 'fr', name: 'French' },
      { code: 'de', name: 'German' },
      { code: 'zh', name: 'Chinese' },
      { code: 'ja', name: 'Japanese' },
      { code: 'ru', name: 'Russian' },
      { code: 'it', name: 'Italian' },
      { code: 'pt', name: 'Portuguese' },
      { code: 'nl', name: 'Dutch' },
      { code: 'pl', name: 'Polish' },
      { code: 'cs', name: 'Czech' },
      { code: 'da', name: 'Danish' },
      { code: 'fi', name: 'Finnish' },
      { code: 'el', name: 'Greek' },
      { code: 'hu', name: 'Hungarian' },
      { code: 'ro', name: 'Romanian' },
      { code: 'sv', name: 'Swedish' }
    ];
  }

  /**
   * Check if translation service is properly configured
   * @returns {boolean} - Whether the service is ready to use
   */
  isConfigured() {
    return !!this.apiKey;
  }
}

// Export a singleton instance
const translationService = new TranslationService();
export default translationService;