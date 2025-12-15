"""
Extract metadata and save to database using Docker backend.
This script triggers re-indexing via the backend API.
"""
import requests
import time
import sys

BACKEND_URL = "http://localhost:8080"

def trigger_reindex(filename):
    """Trigger re-indexing via API."""
    print(f"Triggering re-index for: {filename}")
    
    # Call the re-index endpoint
    response = requests.post(
        f"{BACKEND_URL}/reindex",
        json={"filename": filename}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Re-index triggered: {result}")
        return True
    else:
        print(f"❌ Failed to trigger re-index: {response.status_code}")
        print(f"Response: {response.text}")
        return False


def check_metadata(filename):
    """Check if metadata exists for a document."""
    print(f"\nChecking metadata for: {filename}")
    
    response = requests.get(f"{BACKEND_URL}/documents")
    
    if response.status_code == 200:
        documents = response.json()
        for doc in documents:
            if doc.get("filename") == filename:
                metadata = doc.get("document_metadata")
                if metadata:
                    print(f"✅ Metadata found!")
                    print(f"  Document Type: {metadata.get('document_type')}")
                    print(f"  Subject Name: {metadata.get('subject_name')}")
                    print(f"  Subject Type: {metadata.get('subject_type')}")
                    print(f"  Confidence: {metadata.get('confidence')}")
                    print(f"  Topics: {metadata.get('topics')}")
                    print(f"  Summary: {metadata.get('summary')}")
                    return metadata
                else:
                    print(f"❌ No metadata found")
                    return None
    else:
        print(f"❌ Failed to get documents: {response.status_code}")
        return None


if __name__ == "__main__":
    filename = "Felix Yang- Resume - 2025.pdf"
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    
    print(f"=== Re-indexing {filename} with metadata extraction ===\n")
    
    # Trigger re-index
    if trigger_reindex(filename):
        print("\nWaiting 60 seconds for indexing to complete...")
        time.sleep(60)
        
        # Check metadata
        check_metadata(filename)
    else:
        print("Failed to trigger re-index")
        sys.exit(1)

