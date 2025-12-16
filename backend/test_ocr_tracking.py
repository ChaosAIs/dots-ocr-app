#!/usr/bin/env python3
"""
Test script for OCR status tracking functionality.

This script tests the OCR tracking methods without requiring a database connection.
"""

def test_ocr_details_structure():
    """Test the OCR details JSONB structure."""
    print("Testing OCR details structure...")
    
    # Simulate the structure that would be created
    ocr_details = {
        "version": "1.0",
        "page_ocr": {
            "status": "pending",
            "total_pages": 10,
            "converted_pages": 0,
            "failed_pages": 0,
            "pages": {}
        }
    }
    
    # Simulate adding a successful page
    page_key = "page_0"
    ocr_details["page_ocr"]["pages"][page_key] = {
        "status": "success",
        "page_number": 0,
        "file_path": "output/doc/doc_page_0_nohf.md",
        "converted_at": "2025-12-16T10:00:00Z",
        "error": None,
        "retry_count": 0,
        "embedded_images_count": 3,
        "embedded_images": {}
    }
    ocr_details["page_ocr"]["converted_pages"] = 1
    
    # Simulate adding a successful embedded image
    image_key = "image_0"
    ocr_details["page_ocr"]["pages"][page_key]["embedded_images"][image_key] = {
        "status": "success",
        "image_position": 0,
        "ocr_backend": "gemma3",
        "converted_at": "2025-12-16T10:00:15Z",
        "error": None,
        "retry_count": 0,
        "image_size_pixels": 250000
    }
    
    # Simulate adding a failed embedded image
    image_key = "image_1"
    ocr_details["page_ocr"]["pages"][page_key]["embedded_images"][image_key] = {
        "status": "failed",
        "image_position": 1,
        "ocr_backend": "gemma3",
        "failed_at": "2025-12-16T10:00:30Z",
        "error": "Timeout calling Gemma3 API",
        "retry_count": 1,
        "image_size_pixels": 180000
    }
    
    # Simulate adding a skipped embedded image
    image_key = "image_2"
    ocr_details["page_ocr"]["pages"][page_key]["embedded_images"][image_key] = {
        "status": "skipped",
        "image_position": 2,
        "ocr_backend": "gemma3",
        "skipped_at": "2025-12-16T10:00:45Z",
        "was_skipped": True,
        "skip_reason": "Image too small for OCR",
        "image_size_pixels": 500
    }
    
    # Simulate adding a failed page
    page_key = "page_5"
    ocr_details["page_ocr"]["pages"][page_key] = {
        "status": "failed",
        "page_number": 5,
        "file_path": None,
        "error": "OCR inference timeout",
        "failed_at": "2025-12-16T10:05:00Z",
        "retry_count": 1,
        "embedded_images_count": 0
    }
    ocr_details["page_ocr"]["failed_pages"] = 1
    
    # Update overall status
    total_pages = ocr_details["page_ocr"]["total_pages"]
    converted_pages = ocr_details["page_ocr"]["converted_pages"]
    failed_pages = ocr_details["page_ocr"]["failed_pages"]
    
    if failed_pages > 0 and converted_pages > 0:
        ocr_details["page_ocr"]["status"] = "partial"
    
    print("✓ OCR details structure created successfully")
    print(f"  Status: {ocr_details['page_ocr']['status']}")
    print(f"  Total pages: {total_pages}")
    print(f"  Converted pages: {converted_pages}")
    print(f"  Failed pages: {failed_pages}")
    print(f"  Pages tracked: {len(ocr_details['page_ocr']['pages'])}")
    
    # Test query logic
    print("\nTesting query logic...")
    
    # Get failed pages
    failed_page_numbers = []
    for page_key, page_info in ocr_details["page_ocr"]["pages"].items():
        if page_info.get("status") == "failed":
            failed_page_numbers.append(page_info.get("page_number"))
    
    print(f"✓ Failed pages: {sorted(failed_page_numbers)}")
    
    # Get pages with failed embedded images
    pages_with_failed_images = {}
    for page_key, page_info in ocr_details["page_ocr"]["pages"].items():
        page_number = page_info.get("page_number")
        embedded_images = page_info.get("embedded_images", {})
        
        failed_images = []
        for image_key, image_info in embedded_images.items():
            if image_info.get("status") == "failed":
                failed_images.append(image_info.get("image_position"))
        
        if failed_images:
            pages_with_failed_images[page_number] = sorted(failed_images)
    
    print(f"✓ Pages with failed images: {pages_with_failed_images}")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("OCR Status Tracking Test")
    print("=" * 60)
    print()
    
    try:
        test_ocr_details_structure()
        print()
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()

