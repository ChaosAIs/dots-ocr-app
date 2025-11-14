#!/usr/bin/env python3
"""
Direct test of DeepSeek OCR API to debug the empty output issue.
"""

import base64
import re
from pathlib import Path
from openai import OpenAI

def format_grounding_output(content: str) -> str:
    """Format grounding mode output with intelligent label-value pairing."""
    # Split into lines and clean
    lines = [line.strip() for line in content.split('\n') if line.strip()]

    # Collect percentage values that need pairing
    # Looking at the raw output: 79%, 30%, 12% appear before Gross profit, Operating profit margin, Net profit margin
    percentage_buffer = []

    # Intelligent pairing and formatting
    formatted_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Detect section headers
        if any(keyword in line.lower() for keyword in ['report', 'financial', 'analysis', 'breakdown', 'position', 'opex', 'forecast', 'revenue', 'costs']):
            if len(line) < 60 and not any(c in line for c in ['$', '%']) and ':' not in line:
                formatted_lines.append(f"\n## {line}\n")
                i += 1
                continue

        # Collect standalone percentages into buffer
        if re.match(r'^\d+%$', line):
            percentage_buffer.append(line)
            i += 1
            continue

        # Check if this line is a label that should be paired with a buffered percentage
        # Labels like "Gross profit", "Operating profit margin", "Net profit margin"
        if percentage_buffer and 'profit' in line.lower() or 'margin' in line.lower():
            if percentage_buffer:
                value = percentage_buffer.pop(0)
                formatted_lines.append(f"- **{line}**: {value}")
                i += 1
                continue

        # Handle cash flow items (IN/OUT with amounts)
        if line in ['IN', 'OUT'] and i + 1 < len(lines) and '$' in lines[i + 1]:
            formatted_lines.append(f"- **Cash {line}**: {lines[i + 1]}")
            i += 2
            continue

        # Handle pie chart labels (product categories)
        if line in ['Desktops', 'Portables', 'iPod', 'Accessories']:
            formatted_lines.append(f"- {line}")
            i += 1
            continue

        # Default: just add the line
        if '$' in line or '%' in line:
            formatted_lines.append(f"- **{line}**")
        else:
            formatted_lines.append(f"- {line}")

        i += 1

    return '\n'.join(formatted_lines)

def encode_image_to_base64(image_path: Path) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_deepseek_ocr():
    """Test DeepSeek OCR with different prompts."""
    
    # Initialize client
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8005/v1"
    )
    
    # Find the test image
    image_path = Path("input/One-Pager-Business-Monthly-Financial-Report.png")
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"✅ Found image: {image_path}")
    
    # Encode image
    base64_image = encode_image_to_base64(image_path)
    print(f"✅ Encoded image to base64 ({len(base64_image)} chars)")
    
    # Test grounding mode with markdown formatting
    test_configs = [
        {
            "name": "Grounding Mode with Markdown Formatting",
            "system": None,
            "prompt": "<|grounding|>Extract all text and data from this dashboard.",
            "max_tokens": 4096,
            "remove_bboxes": True
        },
    ]
    
    for config in test_configs:
        print(f"\n{'='*80}")
        print(f"Testing: {config['name']}")
        if config['system']:
            print(f"System: {config['system'][:100]}...")
        print(f"Prompt: {config['prompt'][:100]}...")
        print(f"{'='*80}")

        try:
            # Prepare messages
            messages = []
            if config['system']:
                messages.append({
                    "role": "system",
                    "content": config['system']
                })

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": config['prompt']
                    }
                ]
            })
            
            # Call API
            print("📞 Calling DeepSeek OCR API...")
            api_params = {
                "model": "deepseek-ai/DeepSeek-OCR",
                "messages": messages,
                "max_tokens": config.get('max_tokens', 4096),
                "temperature": 0.0,
                "top_p": 0.95,
            }

            # Add stop sequences if specified
            if 'stop' in config:
                api_params['stop'] = config['stop']
                print(f"   Using stop sequences: {config['stop']}")

            response = client.chat.completions.create(**api_params)
            
            # Extract result
            content = response.choices[0].message.content

            # Remove bounding boxes if requested
            if config.get('remove_bboxes', False):
                import re
                original_length = len(content)

                # Remove bounding box coordinates like [[x1, y1, x2, y2]]
                content = re.sub(r'\[\[\d+,\s*\d+,\s*\d+,\s*\d+\]\]', '', content)

                # Remove grounding tags like <|ref|>text<|/ref|><|det|><|/det|>
                content = re.sub(r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>.*?<\|/det\|>', r'\1\n', content)

                print(f"   Removed bounding boxes/tags: {original_length} -> {len(content)} chars")

                # Apply intelligent formatting
                content = format_grounding_output(content)

            print(f"✅ Response received!")
            print(f"   Length: {len(content)} characters")

            # Count "Add text here" occurrences
            placeholder_count = content.count("Add text here")
            print(f"   'Add text here' count: {placeholder_count}")

            print(f"\n{'='*80}")
            print("FULL OUTPUT:")
            print(f"{'='*80}")
            print(content)
            print(f"{'='*80}")

            if len(content) == 0:
                print("❌ EMPTY OUTPUT!")
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_deepseek_ocr()

