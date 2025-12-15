/**
 * Content Personalization Utilities
 *
 * This module provides functions for personalizing content based on user preferences
 * and profile in the Physical AI & Humanoid Robotics book.
 */

/**
 * Filter content based on user's experience level
 * @param {Array} contentList - List of content items
 * @param {string} experienceLevel - User's experience level (beginner, intermediate, advanced)
 * @returns {Array} - Filtered content list
 */
export function filterByExperienceLevel(contentList, experienceLevel) {
  if (!contentList || !Array.isArray(contentList)) {
    return [];
  }

  // Define difficulty thresholds
  const difficultyThresholds = {
    beginner: ['beginner'],
    intermediate: ['beginner', 'intermediate'],
    advanced: ['beginner', 'intermediate', 'advanced']
  };

  const allowedDifficulties = difficultyThresholds[experienceLevel] || ['beginner'];

  return contentList.filter(item => {
    // If no difficulty specified, show to all levels
    if (!item.difficulty) {
      return true;
    }

    return allowedDifficulties.includes(item.difficulty.toLowerCase());
  });
}

/**
 * Filter content based on user's interests
 * @param {Array} contentList - List of content items
 * @param {Array} interests - User's selected interests
 * @returns {Array} - Filtered content list
 */
export function filterByInterests(contentList, interests) {
  if (!contentList || !Array.isArray(contentList) || !interests || !Array.isArray(interests)) {
    return contentList || [];
  }

  // If no interests selected, show all content
  if (interests.length === 0) {
    return contentList;
  }

  return contentList.filter(item => {
    // If no tags specified, show to all users
    if (!item.tags || !Array.isArray(item.tags)) {
      return true;
    }

    // Check if any of the item's tags match user's interests
    return item.tags.some(tag =>
      interests.some(interest =>
        tag.toLowerCase().includes(interest.toLowerCase()) ||
        interest.toLowerCase().includes(tag.toLowerCase())
      )
    );
  });
}

/**
 * Sort content based on relevance to user profile
 * @param {Array} contentList - List of content items
 * @param {Object} userProfile - User's profile with interests and experience
 * @returns {Array} - Sorted content list
 */
export function sortContentByRelevance(contentList, userProfile) {
  if (!contentList || !Array.isArray(contentList) || !userProfile) {
    return contentList || [];
  }

  return [...contentList].sort((a, b) => {
    const aRelevance = calculateRelevanceScore(a, userProfile);
    const bRelevance = calculateRelevanceScore(b, userProfile);

    // Sort by relevance score (descending)
    return bRelevance - aRelevance;
  });
}

/**
 * Calculate relevance score for a content item based on user profile
 * @param {Object} content - Content item
 * @param {Object} userProfile - User's profile
 * @returns {number} - Relevance score (0-1)
 */
function calculateRelevanceScore(content, userProfile) {
  let score = 0;

  // Check interest match (weight: 0.5)
  if (content.tags && userProfile.interests && Array.isArray(content.tags) && Array.isArray(userProfile.interests)) {
    const matchingTags = content.tags.filter(tag =>
      userProfile.interests.some(interest =>
        tag.toLowerCase().includes(interest.toLowerCase()) ||
        interest.toLowerCase().includes(tag.toLowerCase())
      )
    );

    if (matchingTags.length > 0) {
      score += (matchingTags.length / content.tags.length) * 0.5;
    }
  }

  // Check experience level match (weight: 0.3)
  if (content.difficulty && userProfile.experienceLevel) {
    const difficultyMatch = checkDifficultyMatch(content.difficulty, userProfile.experienceLevel);
    score += difficultyMatch * 0.3;
  }

  // Check other factors like completion status, recency, etc. (weight: 0.2)
  // This could include factors like:
  // - Whether user has already completed this content
  // - How recently the content was published
  // - User's past interactions with similar content

  // Ensure score is between 0 and 1
  return Math.min(score, 1);
}

/**
 * Check how well content difficulty matches user experience level
 * @param {string} contentDifficulty - Content difficulty level
 * @param {string} userExperience - User's experience level
 * @returns {number} - Match score (0-1)
 */
function checkDifficultyMatch(contentDifficulty, userExperience) {
  const matchMatrix = {
    beginner: { beginner: 1.0, intermediate: 0.7, advanced: 0.3 },
    intermediate: { beginner: 0.5, intermediate: 1.0, advanced: 0.8 },
    advanced: { beginner: 0.2, intermediate: 0.5, advanced: 1.0 }
  };

  const level = contentDifficulty.toLowerCase();
  const experience = userExperience.toLowerCase();

  return matchMatrix[experience]?.[level] || 0;
}

/**
 * Personalize content list based on user profile
 * @param {Array} contentList - List of content items
 * @param {Object} userProfile - User's profile
 * @param {Object} options - Personalization options
 * @returns {Array} - Personalized content list
 */
export function personalizeContent(contentList, userProfile, options = {}) {
  const {
    filterByLevel = true,
    filterByInterests = true,
    sortByRelevance = true
  } = options;

  let result = [...contentList];

  if (filterByLevel) {
    result = filterByExperienceLevel(result, userProfile.experienceLevel);
  }

  if (filterByInterests) {
    result = filterByInterests(result, userProfile.interests);
  }

  if (sortByRelevance) {
    result = sortContentByRelevance(result, userProfile);
  }

  return result;
}

/**
 * Get personalized recommendations based on user profile
 * @param {Array} allContent - Complete content catalog
 * @param {Object} userProfile - User's profile
 * @param {number} limit - Maximum number of recommendations
 * @returns {Array} - Recommended content items
 */
export function getPersonalizedRecommendations(allContent, userProfile, limit = 5) {
  // Filter and sort all content
  const personalizedContent = personalizeContent(allContent, userProfile);

  // Take only the specified number of items
  return personalizedContent.slice(0, limit);
}

/**
 * Highlight content based on user preferences
 * @param {string} content - Content string
 * @param {Object} userProfile - User's profile
 * @returns {string} - Content with highlighted relevant terms
 */
export function highlightContentByPreferences(content, userProfile) {
  if (!content || typeof content !== 'string' || !userProfile?.interests) {
    return content;
  }

  let highlightedContent = content;

  // Highlight user interests in the content
  userProfile.interests.forEach(interest => {
    if (interest) {
      // Create a case-insensitive regex to find the interest
      const regex = new RegExp(`(${interest})`, 'gi');
      highlightedContent = highlightedContent.replace(regex, '<mark class="highlight-personalized">$1</mark>');
    }
  });

  return highlightedContent;
}