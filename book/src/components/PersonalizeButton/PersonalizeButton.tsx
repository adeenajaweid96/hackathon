import React, { useState } from 'react';
import { useColorMode } from '@docusaurus/theme-common';
import BrowserOnly from '@docusaurus/BrowserOnly';

const PersonalizeButton = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [userProfile, setUserProfile] = useState({
    programmingExperience: 'beginner',
    hardwareAccess: ['None'],
    roboticsBackground: 'beginner',
    learningGoals: []
  });
  const { colorMode } = useColorMode();

  const toggleModal = () => {
    setIsOpen(!isOpen);
  };

  const handleSavePreferences = () => {
    // In a real implementation, this would save to a backend or local storage
    console.log('Saving user preferences:', userProfile);
    setIsOpen(false);
  };

  const handleInputChange = (field: string, value: any) => {
    setUserProfile(prev => ({
      ...prev,
      [field]: value
    }));
  };

  return (
    <BrowserOnly>
      {() => (
        <div className={`personalize-component ${colorMode}`}>
          <button
            className="personalize-btn"
            onClick={toggleModal}
            aria-label="Personalize your learning experience"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z" />
            </svg>
            Personalize
          </button>

          {isOpen && (
            <div className="personalize-modal-overlay" onClick={toggleModal}>
              <div className="personalize-modal" onClick={(e) => e.stopPropagation()}>
                <div className="personalize-modal-header">
                  <h3>Personalize Your Learning</h3>
                  <button
                    className="personalize-modal-close"
                    onClick={toggleModal}
                    aria-label="Close modal"
                  >
                    ×
                  </button>
                </div>
                <div className="personalize-modal-content">
                  <div className="personalize-form-group">
                    <label htmlFor="programmingExperience">Programming Experience</label>
                    <select
                      id="programmingExperience"
                      value={userProfile.programmingExperience}
                      onChange={(e) => handleInputChange('programmingExperience', e.target.value)}
                    >
                      <option value="beginner">Beginner</option>
                      <option value="intermediate">Intermediate</option>
                      <option value="advanced">Advanced</option>
                    </select>
                  </div>

                  <div className="personalize-form-group">
                    <label>Hardware Access</label>
                    <div className="checkbox-group">
                      <label>
                        <input
                          type="checkbox"
                          checked={userProfile.hardwareAccess.includes('Jetson Orin Nano')}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setUserProfile(prev => ({
                                ...prev,
                                hardwareAccess: [...prev.hardwareAccess, 'Jetson Orin Nano']
                              }));
                            } else {
                              setUserProfile(prev => ({
                                ...prev,
                                hardwareAccess: prev.hardwareAccess.filter(item => item !== 'Jetson Orin Nano')
                              }));
                            }
                          }}
                        /> Jetson Orin Nano
                      </label>
                      <label>
                        <input
                          type="checkbox"
                          checked={userProfile.hardwareAccess.includes('RealSense D435i')}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setUserProfile(prev => ({
                                ...prev,
                                hardwareAccess: [...prev.hardwareAccess, 'RealSense D435i']
                              }));
                            } else {
                              setUserProfile(prev => ({
                                ...prev,
                                hardwareAccess: prev.hardwareAccess.filter(item => item !== 'RealSense D435i')
                              }));
                            }
                          }}
                        /> RealSense D435i
                      </label>
                      <label>
                        <input
                          type="checkbox"
                          checked={userProfile.hardwareAccess.includes('Other')}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setUserProfile(prev => ({
                                ...prev,
                                hardwareAccess: [...prev.hardwareAccess, 'Other']
                              }));
                            } else {
                              setUserProfile(prev => ({
                                ...prev,
                                hardwareAccess: prev.hardwareAccess.filter(item => item !== 'Other')
                              }));
                            }
                          }}
                        /> Other
                      </label>
                    </div>
                  </div>

                  <div className="personalize-form-group">
                    <label htmlFor="roboticsBackground">Robotics Background</label>
                    <select
                      id="roboticsBackground"
                      value={userProfile.roboticsBackground}
                      onChange={(e) => handleInputChange('roboticsBackground', e.target.value)}
                    >
                      <option value="beginner">Beginner</option>
                      <option value="intermediate">Intermediate</option>
                      <option value="advanced">Advanced</option>
                    </select>
                  </div>
                </div>
                <div className="personalize-modal-footer">
                  <button className="personalize-cancel-btn" onClick={toggleModal}>
                    Cancel
                  </button>
                  <button className="personalize-save-btn" onClick={handleSavePreferences}>
                    Save Preferences
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </BrowserOnly>
  );
};

export default PersonalizeButton;