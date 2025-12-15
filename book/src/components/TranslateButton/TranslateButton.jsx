import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import translationService from './translationService';
import './TranslateButton.css';

const TranslateButton = () => {
  const [isTranslating, setIsTranslating] = useState(false);
  const [currentLanguage, setCurrentLanguage] = useState('en');
  const [isOpen, setIsOpen] = useState(false);

  // Supported languages
  const languages = [
    { code: 'en', name: 'English', flag: '🇺🇸' },
    { code: 'ur', name: 'Urdu', flag: '🇵🇰' },
    { code: 'es', name: 'Spanish', flag: '🇪🇸' },
    { code: 'fr', name: 'French', flag: '🇫🇷' },
    { code: 'de', name: 'German', flag: '🇩🇪' },
    { code: 'zh', name: 'Chinese', flag: '🇨🇳' }
  ];

  // Get available languages (excluding current language)
  const availableLanguages = languages.filter(lang => lang.code !== currentLanguage);

  const toggleMenu = () => {
    setIsOpen(!isOpen);
  };

  const selectLanguage = async (languageCode) => {
    if (languageCode === currentLanguage) return;

    setIsTranslating(true);
    setIsOpen(false);

    try {
      // In a real implementation, this would call a translation API
      // For now, we'll just update the language state
      setCurrentLanguage(languageCode);

      // Save selected language to localStorage
      localStorage.setItem('preferredLanguage', languageCode);

      // Simulate translation delay
      await new Promise(resolve => setTimeout(resolve, 500));

      // In a real implementation, we would translate the page content here
      handleTranslation(languageCode);
    } catch (error) {
      console.error('Translation error:', error);
    } finally {
      setIsTranslating(false);
    }
  };

  const handleTranslation = async (languageCode) => {
    // Translate the page content using the translation service
    try {
      await translationService.translatePageContent(languageCode);
      console.log(`Page translated to ${languageCode}`);
    } catch (error) {
      console.error('Translation failed:', error);
      // In case of failure, we could show an error message to the user
    }
  };

  // Load saved language preference on component mount
  useEffect(() => {
    const savedLanguage = localStorage.getItem('preferredLanguage');
    if (savedLanguage) {
      setCurrentLanguage(savedLanguage);
    }
  }, []);

  const currentLanguageInfo = languages.find(lang => lang.code === currentLanguage);

  return (
    <div className="translate-button-container">
      <button
        className={`translate-button ${isTranslating ? 'translating' : ''}`}
        onClick={toggleMenu}
        aria-label="Translate content"
        title="Translate content"
        disabled={isTranslating}
      >
        {isTranslating ? (
          <span className="translating-indicator">🔄</span>
        ) : (
          <>
            <span className="translate-icon">{currentLanguageInfo?.flag || '🌐'}</span>
            <span className="translate-label">Translate</span>
          </>
        )}
      </button>

      {isOpen && (
        <div className="translate-menu">
          <div className="translate-header">
            <h3>Select Language</h3>
            <button
              className="close-button"
              onClick={() => setIsOpen(false)}
              aria-label="Close"
            >
              ×
            </button>
          </div>

          <div className="language-options">
            {availableLanguages.map(language => (
              <button
                key={language.code}
                className="language-option"
                onClick={() => selectLanguage(language.code)}
                disabled={isTranslating}
              >
                <span className="language-flag">{language.flag}</span>
                <span className="language-name">{language.name}</span>
                <span className="language-code">({language.code.toUpperCase()})</span>
              </button>
            ))}
          </div>

          <div className="current-language">
            <p>Current: {currentLanguageInfo?.name} {currentLanguageInfo?.flag}</p>
          </div>
        </div>
      )}
    </div>
  );
};

TranslateButton.propTypes = {
  // No props needed for this component
};

export default TranslateButton;