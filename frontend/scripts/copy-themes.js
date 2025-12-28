/**
 * Script to copy PrimeReact theme CSS files to public/themes folder
 * Run this after npm install to ensure themes are available
 */

const fs = require('fs');
const path = require('path');

const themes = [
  'saga-blue', 'saga-green', 'saga-orange', 'saga-purple',
  'vela-blue', 'vela-green', 'vela-orange', 'vela-purple',
  'arya-blue', 'arya-green', 'arya-orange', 'arya-purple',
  'lara-light-blue', 'lara-light-indigo', 'lara-light-purple', 'lara-light-teal',
  'lara-dark-blue', 'lara-dark-indigo', 'lara-dark-purple', 'lara-dark-teal'
];

const sourceDir = path.join(__dirname, '..', 'node_modules', 'primereact', 'resources', 'themes');
const targetDir = path.join(__dirname, '..', 'public', 'themes');

// Create target directory if it doesn't exist
if (!fs.existsSync(targetDir)) {
  fs.mkdirSync(targetDir, { recursive: true });
}

themes.forEach(theme => {
  const sourceFile = path.join(sourceDir, theme, 'theme.css');
  const targetThemeDir = path.join(targetDir, theme);
  const targetFile = path.join(targetThemeDir, 'theme.css');

  if (fs.existsSync(sourceFile)) {
    // Create theme directory if it doesn't exist
    if (!fs.existsSync(targetThemeDir)) {
      fs.mkdirSync(targetThemeDir, { recursive: true });
    }

    // Copy theme CSS file
    fs.copyFileSync(sourceFile, targetFile);
    console.log(`Copied theme: ${theme}`);
  } else {
    console.warn(`Theme not found: ${theme}`);
  }
});

// Fix lara-light-teal theme - incorrect highlight-bg color in PrimeReact package
// The original value #0f766e (dark teal) should be #f0fdfa (light teal) for light theme
const laraLightTealFile = path.join(targetDir, 'lara-light-teal', 'theme.css');
if (fs.existsSync(laraLightTealFile)) {
  let content = fs.readFileSync(laraLightTealFile, 'utf8');
  content = content.replace('--highlight-bg: #0f766e;', '--highlight-bg: #f0fdfa;');
  fs.writeFileSync(laraLightTealFile, content, 'utf8');
  console.log('Fixed lara-light-teal highlight-bg color');
}

console.log('Theme copy complete!');
