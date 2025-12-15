/**
 * Selected Text Handler for Chatbot Widget
 *
 * This utility handles the functionality for querying selected text
 * from the book content using the RAG chatbot.
 */

class SelectedTextHandler {
  constructor(apiUrl, setMessages, setIsLoading, sessionId) {
    this.apiUrl = apiUrl;
    this.setMessages = setMessages;
    this.setIsLoading = setIsLoading;
    this.getSessionId = () => sessionId;
    this.isActive = false;
  }

  // Initialize the selected text functionality
  init() {
    // Add event listener for mouseup to detect text selection
    document.addEventListener('mouseup', this.handleTextSelection.bind(this));

    // Also add support for touch events on mobile
    document.addEventListener('touchend', this.handleTextSelection.bind(this));

    this.isActive = true;
  }

  // Handle text selection event
  handleTextSelection(event) {
    const selectedText = this.getSelectedText();

    if (selectedText && selectedText.trim().length > 10) { // Only if meaningful text is selected
      this.showQueryDialog(selectedText, event);
    }
  }

  // Get the currently selected text
  getSelectedText() {
    if (window.getSelection) {
      return window.getSelection().toString();
    } else if (document.selection && document.selection.type !== 'Control') {
      return document.selection.createRange().text;
    }
    return '';
  }

  // Show a dialog or tooltip to ask user if they want to query the selected text
  showQueryDialog(selectedText, event) {
    // Create a temporary button or tooltip
    const queryButton = document.createElement('div');
    queryButton.id = 'chatbot-query-button';
    queryButton.innerHTML = 'Ask AI';
    queryButton.style.position = 'fixed';
    queryButton.style.top = (event.clientY + 10) + 'px';
    queryButton.style.left = (event.clientX + 10) + 'px';
    queryButton.style.backgroundColor = '#2563eb';
    queryButton.style.color = 'white';
    queryButton.style.padding = '8px 12px';
    queryButton.style.borderRadius = '4px';
    queryButton.style.cursor = 'pointer';
    queryButton.style.zIndex = '10000';
    queryButton.style.fontSize = '14px';
    queryButton.style.boxShadow = '0 2px 6px rgba(0,0,0,0.2)';
    queryButton.style.fontFamily = 'sans-serif';

    // Add click handler to the button
    queryButton.onclick = () => {
      this.querySelectedText(selectedText);
      document.body.removeChild(queryButton);
    };

    // Remove button after a delay if not clicked
    setTimeout(() => {
      if (document.contains(queryButton)) {
        document.body.removeChild(queryButton);
      }
    }, 3000);

    document.body.appendChild(queryButton);
  }

  // Query the selected text to the backend
  async querySelectedText(selectedText) {
    try {
      // Set loading state
      this.setIsLoading(true);

      // Add a temporary message about the selected text query
      this.addSelectedTextMessage(selectedText);

      // Make API call to backend
      const response = await fetch(`${this.apiUrl}/query-selected-text`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          selected_text: selectedText,
          question: "Explain this concept or provide more details.",
          session_id: this.getSessionId()
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Add the response to the chat
      this.addBotResponse(data);

    } catch (error) {
      console.error('Error querying selected text:', error);
      this.addErrorMessage('Sorry, I encountered an error processing the selected text.');
    } finally {
      // Reset loading state
      this.setIsLoading(false);
    }
  }

  // Add a message about selected text to the chat
  addSelectedTextMessage(selectedText) {
    const message = {
      id: Date.now(),
      text: `Selected text: ${selectedText.substring(0, 100)}${selectedText.length > 100 ? '...' : ''}`,
      sender: 'user',
      isSelection: true,
      timestamp: new Date().toISOString()
    };

    this.setMessages(prev => [...prev, message]);
  }

  // Add bot response to the chat
  addBotResponse(responseData) {
    const message = {
      id: Date.now() + 1,
      text: responseData.response,
      sender: 'bot',
      sources: responseData.sources || [],
      confidence: responseData.confidence,
      timestamp: responseData.timestamp || new Date().toISOString()
    };

    this.setMessages(prev => [...prev, message]);
  }

  // Add error message to the chat
  addErrorMessage(errorText) {
    const message = {
      id: Date.now() + 1,
      text: errorText,
      sender: 'bot',
      isError: true,
      timestamp: new Date().toISOString()
    };

    this.setMessages(prev => [...prev, message]);
  }

  // Clean up event listeners
  destroy() {
    document.removeEventListener('mouseup', this.handleTextSelection.bind(this));
    document.removeEventListener('touchend', this.handleTextSelection.bind(this));
    this.isActive = false;
  }
}

export default SelectedTextHandler;