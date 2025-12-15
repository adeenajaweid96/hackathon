import React, { useState, useEffect } from 'react';
import { useThemeContext } from '@docusaurus/theme-common';
import './ThemeToggle.css';

const ThemeToggle = () => {
  const { isDarkTheme, setLightTheme, setDarkTheme } = useThemeContext();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const toggleTheme = () => {
    if (isDarkTheme) {
      setLightTheme();
    } else {
      setDarkTheme();
    }
  };

  if (!mounted) {
    return (
      <div className="theme-toggle-skeleton">
        <div className="theme-toggle-skeleton-circle"></div>
      </div>
    );
  }

  return (
    <button
      className={`theme-toggle ${isDarkTheme ? 'theme-toggle--dark' : 'theme-toggle--light'}`}
      onClick={toggleTheme}
      aria-label={`Switch to ${isDarkTheme ? 'light' : 'dark'} theme`}
      title={`Switch to ${isDarkTheme ? 'light' : 'dark'} theme`}
      type="button"
    >
      <div className="theme-toggle-icon">
        {isDarkTheme ? '☀️' : '🌙'}
      </div>
    </button>
  );
};

export default ThemeToggle;