# Table of Contents Scroll Fix

## Problem
When clicking on items in the Table of Contents sidebar, the following error occurred:
```
TypeError: document.getElementById is not a function
```

## Root Cause
The issue had two parts:

1. **ID Mismatch**: The table of contents was generating IDs based on line index (`heading-0`, `heading-1`, etc.), but the markdown components were generating IDs based on heading text (`heading-Introduction`, `heading-Page 0`, etc.). These didn't match.

2. **DOM Context Issue**: The markdown content is rendered inside a Dialog component, which creates a separate DOM context. Using `document.getElementById()` was searching the entire document instead of just the content container.

## Solution

### 1. Added useRef Hook
```javascript
import { useState, useEffect, useMemo, useRef } from "react";

const contentRef = useRef(null);
```

### 2. Fixed ID Generation
Updated the table of contents generation to use heading text for IDs:
```javascript
const tableOfContents = useMemo(() => {
  const headings = [];
  const lines = content.split("\n");
  lines.forEach((line) => {
    const match = line.match(/^(#{1,6})\s+(.+)$/);
    if (match) {
      const level = match[1].length;
      const title = match[2];
      // Generate ID from title to match the markdown components
      const id = `heading-${title}`;
      headings.push({ level, title, id });
    }
  });
  return headings;
}, [content]);
```

### 3. Updated scrollToHeading Function
Changed from using `document.getElementById()` to searching within the content container:
```javascript
const scrollToHeading = (id) => {
  // Search for the element within the content container
  if (contentRef.current) {
    const element = contentRef.current.querySelector(`[id="${id}"]`);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  }
};
```

### 4. Attached Ref to Content Container
Added the ref to the markdown-body div:
```javascript
<div className="markdown-body" ref={contentRef}>
  <ReactMarkdown components={markdownComponents}>
    {highlightedContent}
  </ReactMarkdown>
</div>
```

## How It Works Now

1. **ID Generation**: Both the TOC and markdown components now use the same ID scheme: `heading-${title}`
2. **DOM Scoping**: The `scrollToHeading` function searches only within the content container using `querySelector`
3. **Smooth Scrolling**: When a TOC item is clicked, it finds the matching heading and scrolls to it smoothly

## Example Flow

1. User clicks "Page 0" in the Table of Contents
2. `scrollToHeading("heading-Page 0")` is called
3. Function searches within `contentRef.current` for element with `id="heading-Page 0"`
4. Element is found and scrolled into view with smooth animation

## Testing

✅ Build successful with no new errors
✅ TOC items are clickable
✅ Smooth scrolling works correctly
✅ Works with both single-page and multi-page documents
✅ No console errors

## Files Modified

- `frontend/src/components/documents/markdownViewer.jsx`
  - Added `useRef` import
  - Added `contentRef` state
  - Updated `tableOfContents` useMemo to generate consistent IDs
  - Updated `scrollToHeading` function to use ref-based DOM search
  - Added `ref={contentRef}` to markdown-body div

## Backward Compatibility

✅ All existing functionality preserved
✅ No breaking changes
✅ Works with all document types

