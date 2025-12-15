import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import './PersonalizeButton.css';

const PersonalizeButton = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [userProfile, setUserProfile] = useState(() => {
    const savedProfile = localStorage.getItem('userProfile');
    return savedProfile ? JSON.parse(savedProfile) : {
      name: '',
      experienceLevel: 'beginner',
      interests: [],
      preferredLanguage: 'en',
      theme: 'dark',
      notifications: true,
      fontSize: 'medium',
      showAnimations: true,
      learningGoals: []
    };
  });

  // Save profile to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('userProfile', JSON.stringify(userProfile));
  }, [userProfile]);

  const toggleMenu = () => {
    setIsOpen(!isOpen);
  };

  const handleSaveProfile = (e) => {
    e.preventDefault();
    setIsOpen(false);
  };

  const handleInputChange = (field, value) => {
    setUserProfile(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleInterestToggle = (interest) => {
    setUserProfile(prev => {
      const newInterests = prev.interests.includes(interest)
        ? prev.interests.filter(i => i !== interest)
        : [...prev.interests, interest];
      return { ...prev, interests: newInterests };
    });
  };

  const interests = [
    'Physical AI', 'ROS 2', 'Humanoid Robotics',
    'Computer Vision', 'Machine Learning', 'Path Planning',
    'Manipulation', 'Human-Robot Interaction'
  ];

  return (
    <div className="personalize-button-container">
      <button
        className="personalize-button"
        onClick={toggleMenu}
        aria-label="Personalize your experience"
        title="Personalize your experience"
      >
        <span className="personalize-icon">👤</span>
        <span className="personalize-label">Personalize</span>
      </button>

      {isOpen && (
        <div className="personalize-menu">
          <div className="personalize-header">
            <h3>Personalize Your Experience</h3>
            <button
              className="close-button"
              onClick={() => setIsOpen(false)}
              aria-label="Close"
            >
              ×
            </button>
          </div>

          <form onSubmit={handleSaveProfile} className="personalize-form">
            <div className="form-group">
              <label htmlFor="name">Name:</label>
              <input
                type="text"
                id="name"
                value={userProfile.name}
                onChange={(e) => handleInputChange('name', e.target.value)}
                placeholder="Enter your name"
              />
            </div>

            <div className="form-group">
              <label>Experience Level:</label>
              <div className="radio-group">
                {['beginner', 'intermediate', 'advanced'].map(level => (
                  <label key={level} className="radio-option">
                    <input
                      type="radio"
                      name="experienceLevel"
                      checked={userProfile.experienceLevel === level}
                      onChange={() => handleInputChange('experienceLevel', level)}
                    />
                    <span className="radio-label">{level.charAt(0).toUpperCase() + level.slice(1)}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="form-group">
              <label>Interests:</label>
              <div className="checkbox-group">
                {interests.map(interest => (
                  <label key={interest} className="checkbox-option">
                    <input
                      type="checkbox"
                      checked={userProfile.interests.includes(interest)}
                      onChange={() => handleInterestToggle(interest)}
                    />
                    <span className="checkbox-label">{interest}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="form-group">
              <label>Font Size:</label>
              <div className="radio-group">
                {['small', 'medium', 'large'].map(size => (
                  <label key={size} className="radio-option">
                    <input
                      type="radio"
                      name="fontSize"
                      checked={userProfile.fontSize === size}
                      onChange={() => handleInputChange('fontSize', size)}
                    />
                    <span className="radio-label">{size.charAt(0).toUpperCase() + size.slice(1)}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="form-group">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={userProfile.notifications}
                  onChange={(e) => handleInputChange('notifications', e.target.checked)}
                />
                <span className="checkbox-text">Enable notifications</span>
              </label>
            </div>

            <div className="form-group">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={userProfile.showAnimations}
                  onChange={(e) => handleInputChange('showAnimations', e.target.checked)}
                />
                <span className="checkbox-text">Show animations</span>
              </label>
            </div>

            <div className="form-actions">
              <button type="button" className="cancel-button" onClick={() => setIsOpen(false)}>
                Cancel
              </button>
              <button type="submit" className="save-button">
                Save Preferences
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
};

PersonalizeButton.propTypes = {
  // No props needed for this component
};

export default PersonalizeButton;