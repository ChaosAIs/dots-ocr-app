# DeepSeek OCR Prompt Engineering - Lessons Learned

## Critical Finding: SIMPLE Prompts Work Best

### Problem Discovered

When trying to improve OCR intelligence with detailed, multi-step prompts, the model produced **worse results**:

**Complex Prompt Example** (FAILED):
```
Analyze this business dashboard or financial report and provide:
1. What is the main purpose of this dashboard?
2. Identify and extract data from ALL charts and visualizations:
   - For each chart: type, title, and ALL data values
   - Extract percentages, amounts, dates, and labels
   - IGNORE placeholder text like "Add text here"
3. Extract key metrics and KPIs shown
4. Identify time periods covered
5. Analyze trends: What's increasing? Decreasing?
6. Provide insights about performance
7. Convert to structured markdown with:
   - Summary section with key findings
   - Separate section for each chart
   - Tables for numerical data
```

**Result**: Model output repetitive generic text:
```
1. Use bullet points to summarize key findings and insights
2. Use bullet points to summarize key findings and insights
3. Use bullet points to summarize key findings and insights
...
```

### Solution: Keep It Simple

**Simple Prompt** (WORKS):
```
Extract all text and data from this dashboard, including chart values and percentages. Convert to markdown.
```

**Result**: Model extracts actual data from the image.

## Why Complex Prompts Fail

1. **Cognitive Overload**: Too many instructions confuse the model
2. **Generic Responses**: Model falls back to template-like output
3. **Lost Focus**: Model focuses on the instructions rather than the image
4. **Repetition**: Model repeats the instruction format instead of extracting data

## Prompt Engineering Best Practices

### ✅ DO: Keep Prompts Short and Direct

**Good Examples**:
- `"Extract all text from this receipt. Convert to markdown."`
- `"Extract text and data from this chart. Convert to markdown."`
- `"Convert this table to markdown format."`
- `"Extract all text from this flowchart and describe the flow."`

### ❌ DON'T: Use Multi-Step Instructions

**Bad Examples**:
- Numbered lists with 5+ steps
- Nested sub-instructions
- Multiple "provide", "analyze", "identify" requests
- Complex conditional logic

### ✅ DO: Be Specific About What to Extract

**Good**:
- `"Extract all text and data from this dashboard, including chart values and percentages."`
- `"Extract all text from this receipt including merchant, date, items, prices, and totals."`

### ❌ DON'T: Ask for Analysis and Insights

**Bad**:
- `"Analyze trends and provide insights"`
- `"What conclusions can be drawn?"`
- `"Explain the overall purpose"`

The model is for OCR (text extraction), not analysis. Keep it focused on extraction.

## Tested Prompts That Work

### Receipt
```
Extract all text from this receipt including merchant, date, items, prices, and totals. Convert to markdown.
```

### Form
```
Extract all form fields, labels, and values from this form. Convert to markdown.
```

### Flowchart
```
Extract all text from this flowchart and describe the flow, further, convert the flowchart to ascii flow diagram.
```
*(This one is slightly longer but has been tested and works well)*

### Diagram
```
Extract all text and labels from this diagram. Describe the components and their relationships. Convert to markdown.
```

### Chart
```
Extract all text, labels, and data values from this chart. Describe the data and trends. Convert to markdown.
```

### Dashboard/Report
```
Extract all text and data from this dashboard, including chart values and percentages. Convert to markdown.
```

### Infographic
```
Extract all text, statistics, and data from this infographic. Convert to markdown preserving the structure.
```

### Table
```
Convert this table to markdown format. Preserve the table structure, headers, and all cell contents accurately.
```

### Screenshot
```
Extract all visible text, labels, and UI elements from this screenshot. Convert to markdown.
```

### Document (Default)
```
Convert the document to markdown format, preserving the structure and formatting.
```

## Key Takeaways

1. **Shorter is Better**: 1-2 sentences max
2. **Direct Instructions**: "Extract X. Convert to Y."
3. **Avoid Analysis**: Focus on extraction, not interpretation
4. **Test Thoroughly**: Always test prompts with real images
5. **Watch for Repetition**: If output is repetitive, simplify the prompt

## Reference

This aligns with the original documentation in the codebase:

> **Critical Findings**:
> 1. Do NOT use `<image>` token
> 2. Do NOT use `<|grounding|>` prefix
> 3. **Keep prompts SHORT and simple**: Long, complex prompts (>100 chars) often cause empty responses
> 4. **Use plain English prompts**: Simple, direct instructions work best

## Conclusion

When in doubt, **keep it simple**. The DeepSeek OCR model works best with clear, concise, direct instructions focused on text extraction, not analysis or interpretation.

