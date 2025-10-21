# Markdown Viewer - Feature Quick Reference

## 🎯 Key Features at a Glance

### 1. Image Lightbox 🖼️
**What it does:** Click any image to view in fullscreen with zoom controls
**How to use:** Click on an image in the markdown
**Keyboard:** Arrow keys to navigate, ESC to close

### 2. Table of Contents 📑
**What it does:** Auto-generated navigation from headings
**How to use:** Click menu icon (☰) to toggle, click heading to scroll
**Supports:** H1 through H6 headings with proper hierarchy

### 3. Search 🔍
**What it does:** Find text in document with highlighting
**How to use:** Type in search box, matches highlight in yellow
**Features:** Case-insensitive, real-time, clear button

### 4. Syntax Highlighting 💻
**What it does:** Professional code highlighting with language detection
**How to use:** Automatic for code blocks with language specified
**Theme:** Atom One Dark (professional dark theme)

### 5. Copy Code 📋
**What it does:** Copy code block to clipboard with one click
**How to use:** Click copy icon on code block
**Feedback:** Toast notification confirms copy

### 6. Professional UI 🎨
**What it does:** Modern, responsive interface
**Features:** Gradient header, custom scrollbars, smooth animations
**Responsive:** Works on desktop, tablet, mobile

## 📊 Component Layout

```
┌─────────────────────────────────────────────────────┐
│  [☰] Filename                          [⬇] [✕]    │  ← Header
├──────────────┬──────────────────────────────────────┤
│              │  [Search box]                        │
│ Table of     ├──────────────────────────────────────┤
│ Contents     │                                      │
│              │  Markdown Content                    │
│ • Heading 1  │  • Code blocks with syntax highlight│
│   • Heading 2│  • Images (clickable)               │
│   • Heading 3│  • Tables                           │
│              │  • Lists                            │
│              │  • Text with search highlights      │
└──────────────┴──────────────────────────────────────┘
```

## 🎮 User Interactions

### Viewing Images
1. Scroll through document
2. Find image
3. Click image
4. Lightbox opens with zoom controls
5. Use arrow keys or buttons to navigate
6. Click outside or press ESC to close

### Using Table of Contents
1. Click menu icon (☰) in header
2. Sidebar appears with all headings
3. Click any heading to scroll to it
4. Click menu icon again to hide sidebar

### Searching Document
1. Type in search box
2. All matches highlight in yellow
3. Scroll to see all matches
4. Click X to clear search
5. Search is case-insensitive

### Copying Code
1. Hover over code block
2. Language label appears
3. Click copy icon
4. Toast shows "Code copied to clipboard"
5. Paste anywhere with Ctrl+V (or Cmd+V)

## 🎨 Visual Design

### Color Scheme
- **Header:** Purple gradient (#667eea to #764ba2)
- **Sidebar:** Light gray (#f8f9fa)
- **Code:** Dark theme (#282c34)
- **Search highlight:** Yellow (#fff3cd)
- **Links:** Blue (#0366d6)

### Typography
- **Headings:** Bold, hierarchical sizing
- **Body:** 16px, 1.6 line-height
- **Code:** Monospace font
- **Labels:** Small, uppercase

### Spacing
- **Padding:** 20-30px for content
- **Margins:** 16px between elements
- **Gap:** 8-10px between components

## 📱 Responsive Behavior

**Desktop (1200px+)**
- Full sidebar visible
- Large images
- Full-width content

**Tablet (768px-1199px)**
- Collapsible sidebar
- Medium images
- Adjusted padding

**Mobile (< 768px)**
- Sidebar as overlay
- Smaller images
- Compact layout

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| ESC | Close lightbox |
| ← → | Navigate images in lightbox |
| Ctrl+C | Copy selected code |
| Ctrl+F | Browser find (works with search) |

## 🔧 Technical Details

### Dependencies
- `react-syntax-highlighter` - Code highlighting
- `yet-another-react-lightbox` - Image viewer
- `primereact` - UI components

### State Variables
- `searchTerm` - Current search text
- `showTOC` - TOC visibility
- `lightboxOpen` - Lightbox visibility
- `lightboxIndex` - Current image index
- `images` - Extracted images array

### Performance
- Memoized TOC generation
- Lazy image extraction
- Efficient search highlighting
- Optimized re-renders

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Images not clickable | Check image paths in markdown |
| TOC empty | Ensure document has headings |
| Search not working | Clear and try again |
| Code not highlighting | Specify language in code block |
| Lightbox not opening | Check browser console for errors |

## 📈 Usage Statistics

- **Code blocks:** Supports all major languages
- **Images:** Unlimited (performance depends on size)
- **Headings:** Up to 6 levels (H1-H6)
- **Search:** Real-time, case-insensitive
- **File size:** Optimized for production

## ✨ Best Practices

1. **For Images:** Use descriptive alt text
2. **For Code:** Always specify language
3. **For Headings:** Use proper hierarchy (H1 → H2 → H3)
4. **For Search:** Use specific terms
5. **For TOC:** Keep heading structure logical

## 🚀 Performance Tips

- Keep images under 5MB
- Use appropriate image formats (PNG, JPG, WebP)
- Limit document length for optimal performance
- Use proper heading hierarchy
- Specify language for code blocks

## 📞 Support

For issues or feature requests, check:
1. Browser console for errors
2. Markdown syntax validity
3. Image file accessibility
4. Component props configuration

---

**Version:** 1.0.0
**Last Updated:** 2025-10-21
**Status:** Production Ready ✅

