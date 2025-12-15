import React, { useState, useEffect, useRef } from 'react';
import { useColorMode } from '@docusaurus/theme-common';
import BrowserOnly from '@docusaurus/BrowserOnly';

const ChatbotWidget = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<{id: number, text: string, sender: 'user' | 'bot'}[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<null | HTMLDivElement>(null);
  const { colorMode } = useColorMode();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user' as const
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Simulate API call to backend
      // In a real implementation, this would call the backend API
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputValue,
          context: 'physical-ai-humanoid-robotics-book'
        }),
      });

      const data = await response.json();
      const botMessage = {
        id: Date.now() + 1,
        text: data.response || "I'm still learning about Physical AI and Humanoid Robotics. Could you rephrase your question?",
        sender: 'bot' as const
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        text: "Sorry, I'm having trouble connecting to the knowledge base right now.",
        sender: 'bot' as const
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  return (
    <BrowserOnly>
      {() => (
        <div className={`chatbot-widget ${colorMode}`}>
          {!isOpen ? (
            <button
              className="chatbot-toggle-btn"
              onClick={toggleChat}
              aria-label="Open chatbot"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
              </svg>
            </button>
          ) : (
            <div className="chatbot-container">
              <div className="chatbot-header">
                <h3>Physical AI Assistant</h3>
                <button
                  className="chatbot-close-btn"
                  onClick={toggleChat}
                  aria-label="Close chatbot"
                >
                  ×
                </button>
              </div>
              <div className="chatbot-messages">
                {messages.length === 0 && (
                  <div className="chatbot-welcome">
                    <p>Hello! I'm your Physical AI and Humanoid Robotics assistant.</p>
                    <p>Ask me anything about the course content, and I'll help you find relevant information.</p>
                  </div>
                )}
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`message ${message.sender}`}
                  >
                    <div className="message-content">{message.text}</div>
                  </div>
                ))}
                {isLoading && (
                  <div className="message bot">
                    <div className="message-content">
                      <div className="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
              <form onSubmit={handleSendMessage} className="chatbot-input-form">
                <input
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder="Ask about Physical AI & Humanoid Robotics..."
                  disabled={isLoading}
                />
                <button
                  type="submit"
                  disabled={!inputValue.trim() || isLoading}
                  aria-label="Send message"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="22" y1="2" x2="11" y2="13"></line>
                    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                  </svg>
                </button>
              </form>
            </div>
          )}
        </div>
      )}
    </BrowserOnly>
  );
};

export default ChatbotWidget;