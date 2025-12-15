import React, { useState } from 'react';
import { useColorMode } from '@docusaurus/theme-common';
import BrowserOnly from '@docusaurus/BrowserOnly';

const TranslateButton = () => {
  const [currentLanguage, setCurrentLanguage] = useState('en');
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const { colorMode } = useColorMode();

  const languages = [
    { code: 'en', name: 'English' },
    { code: 'ur', name: 'Urdu' }
  ];

  const toggleDropdown = () => {
    setIsDropdownOpen(!isDropdownOpen);
  };

  const selectLanguage = (langCode: string, langName: string) => {
    // In a real implementation, this would trigger content translation
    console.log(`Switching to language: ${langName} (${langCode})`);
    setCurrentLanguage(langCode);
    setIsDropdownOpen(false);

    // This would typically trigger a content update in a real implementation
    // For now, we'll just log the language change
  };

  return (
    <BrowserOnly>
      {() => (
        <div className={`translate-component ${colorMode}`}>
          <div className="translate-dropdown">
            <button
              className="translate-btn"
              onClick={toggleDropdown}
              aria-label="Change language"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M5 12h14" />
                <path d="M5 12l4-4m-4 4l4 4" />
                <path d="M19 12l-4-4m4 4l-4 4" />
              </svg>
              {languages.find(lang => lang.code === currentLanguage)?.name}
            </button>

            {isDropdownOpen && (
              <div className="translate-dropdown-menu">
                {languages.map((lang) => (
                  <button
                    key={lang.code}
                    className={`translate-dropdown-item ${currentLanguage === lang.code ? 'active' : ''}`}
                    onClick={() => selectLanguage(lang.code, lang.name)}
                  >
                    {lang.name}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </BrowserOnly>
  );
};

export default TranslateButton;