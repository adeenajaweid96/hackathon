import React, { createContext, useContext, useReducer, useEffect } from 'react';

// Initial state for user preferences
const initialState = {
  profile: {
    name: '',
    experienceLevel: 'beginner',
    interests: [],
    preferredLanguage: 'en',
    theme: 'dark',
    notifications: true,
    fontSize: 'medium',
    showAnimations: true,
    learningGoals: []
  },
  isInitialized: false
};

// Action types
const actionTypes = {
  SET_PROFILE: 'SET_PROFILE',
  UPDATE_PREFERENCES: 'UPDATE_PREFERENCES',
  SET_EXPERIENCE_LEVEL: 'SET_EXPERIENCE_LEVEL',
  ADD_INTEREST: 'ADD_INTEREST',
  REMOVE_INTEREST: 'REMOVE_INTEREST',
  SET_THEME: 'SET_THEME',
  SET_FONT_SIZE: 'SET_FONT_SIZE',
  SET_INITIALIZED: 'SET_INITIALIZED'
};

// Reducer function
function userPreferencesReducer(state, action) {
  switch (action.type) {
    case actionTypes.SET_PROFILE:
      return {
        ...state,
        profile: { ...state.profile, ...action.payload },
        isInitialized: true
      };

    case actionTypes.UPDATE_PREFERENCES:
      return {
        ...state,
        profile: { ...state.profile, ...action.payload }
      };

    case actionTypes.SET_EXPERIENCE_LEVEL:
      return {
        ...state,
        profile: { ...state.profile, experienceLevel: action.payload }
      };

    case actionTypes.ADD_INTEREST:
      if (!state.profile.interests.includes(action.payload)) {
        return {
          ...state,
          profile: {
            ...state.profile,
            interests: [...state.profile.interests, action.payload]
          }
        };
      }
      return state;

    case actionTypes.REMOVE_INTEREST:
      return {
        ...state,
        profile: {
          ...state.profile,
          interests: state.profile.interests.filter(interest => interest !== action.payload)
        }
      };

    case actionTypes.SET_THEME:
      return {
        ...state,
        profile: { ...state.profile, theme: action.payload }
      };

    case actionTypes.SET_FONT_SIZE:
      return {
        ...state,
        profile: { ...state.profile, fontSize: action.payload }
      };

    case actionTypes.SET_INITIALIZED:
      return {
        ...state,
        isInitialized: action.payload
      };

    default:
      return state;
  }
}

// Create context
const UserPreferencesContext = createContext();

// Provider component
export function UserPreferencesProvider({ children }) {
  const [state, dispatch] = useReducer(userPreferencesReducer, initialState);

  // Load user preferences from localStorage on mount
  useEffect(() => {
    const savedProfile = localStorage.getItem('userProfile');
    if (savedProfile) {
      try {
        const profile = JSON.parse(savedProfile);
        dispatch({ type: actionTypes.SET_PROFILE, payload: profile });
      } catch (error) {
        console.error('Error loading user profile:', error);
      }
    } else {
      dispatch({ type: actionTypes.SET_INITIALIZED, payload: true });
    }
  }, []);

  // Save user preferences to localStorage whenever they change
  useEffect(() => {
    if (state.isInitialized) {
      localStorage.setItem('userProfile', JSON.stringify(state.profile));

      // Apply theme to document
      document.documentElement.setAttribute('data-theme', state.profile.theme);

      // Apply font size to document
      document.documentElement.style.fontSize =
        state.profile.fontSize === 'small' ? '14px' :
        state.profile.fontSize === 'large' ? '18px' : '16px';
    }
  }, [state.profile, state.isInitialized]);

  // Actions
  const setProfile = (profile) => {
    dispatch({ type: actionTypes.SET_PROFILE, payload: profile });
  };

  const updatePreferences = (preferences) => {
    dispatch({ type: actionTypes.UPDATE_PREFERENCES, payload: preferences });
  };

  const setExperienceLevel = (level) => {
    dispatch({ type: actionTypes.SET_EXPERIENCE_LEVEL, payload: level });
  };

  const addInterest = (interest) => {
    dispatch({ type: actionTypes.ADD_INTEREST, payload: interest });
  };

  const removeInterest = (interest) => {
    dispatch({ type: actionTypes.REMOVE_INTEREST, payload: interest });
  };

  const setTheme = (theme) => {
    dispatch({ type: actionTypes.SET_THEME, payload: theme });
  };

  const setFontSize = (size) => {
    dispatch({ type: actionTypes.SET_FONT_SIZE, payload: size });
  };

  // Function to get personalized content based on user profile
  const getPersonalizedContent = (content, options = {}) => {
    const { experienceLevel, interests } = state.profile;
    const { difficultyFilter = true, interestFilter = true } = options;

    // Filter content based on experience level
    if (difficultyFilter && content.difficulty) {
      if (experienceLevel === 'beginner' && content.difficulty === 'advanced') {
        return null; // Don't show advanced content to beginners
      }
      if (experienceLevel === 'advanced' && content.difficulty === 'beginner') {
        // Optionally show different content or highlight it differently
      }
    }

    // Filter content based on interests
    if (interestFilter && content.tags && interests.length > 0) {
      const hasMatchingInterest = content.tags.some(tag => interests.includes(tag));
      if (!hasMatchingInterest) {
        // Content doesn't match user interests, return with lower priority
        return { ...content, priority: 'low' };
      }
    }

    return content;
  };

  // Function to check if content is relevant to user
  const isContentRelevant = (content) => {
    const { experienceLevel, interests } = state.profile;

    // Check experience level match
    if (content.difficulty) {
      if (experienceLevel === 'beginner' && content.difficulty === 'advanced') {
        return false;
      }
    }

    // Check interest match
    if (content.tags && interests.length > 0) {
      return content.tags.some(tag => interests.includes(tag));
    }

    return true;
  };

  const value = {
    ...state,
    actions: {
      setProfile,
      updatePreferences,
      setExperienceLevel,
      addInterest,
      removeInterest,
      setTheme,
      setFontSize
    },
    utils: {
      getPersonalizedContent,
      isContentRelevant
    }
  };

  return (
    <UserPreferencesContext.Provider value={value}>
      {children}
    </UserPreferencesContext.Provider>
  );
}

// Custom hook to use the context
export function useUserPreferences() {
  const context = useContext(UserPreferencesContext);
  if (!context) {
    throw new Error('useUserPreferences must be used within a UserPreferencesProvider');
  }
  return context;
}

export default UserPreferencesContext;