import React from 'react';

// Simple test to verify React is working
test('React is working', () => {
  const element = React.createElement('div', null, 'Hello React');
  expect(element.type).toBe('div');
  expect(element.props.children).toBe('Hello React');
});

// Test that the App component can be imported
test('App component can be imported', () => {
  const App = require('./App').default;
  expect(typeof App).toBe('function');
});
