# Quick Start Guide - Document Upload and Conversion

## Getting Started

### Prerequisites

- Backend running on port 8080 (or configured API_DOMAIN)
- Frontend running on port 3000
- Python vLLM service running for OCR processing

### Starting the Application

1. **Start the Backend:**

```bash
cd backend
python main.py --port 8080
```

2. **Start the Frontend:**

```bash
cd frontend
npm start
```

3. **Access the Application:**
   Open browser and navigate to `http://localhost:3000/docs`

## Using the Document Upload Feature

### Step 1: Upload a Document

1. Navigate to the Home page
2. In the "Upload Documents" section, you can:

   - Click "Choose Files" button to select files
   - Or drag and drop files directly into the upload area

3. Supported file types:

   - PDF (.pdf)
   - Images (.png, .jpg, .jpeg, .gif, .bmp)
   - Documents (.doc, .docx)
   - Spreadsheets (.xls, .xlsx)

4. Maximum file size: 50MB

5. Click "Upload" to upload the selected files

### Step 2: View Uploaded Documents

1. After upload, the document appears in the "Uploaded Documents" table
2. The table shows:
   - Filename
   - File size
   - Upload time
   - Conversion status (Converted/Pending)
   - Action buttons

### Step 3: Convert Document to Markdown

**For PDF and Image files:**

- If not already converted, click the refresh button (⟳) in the Actions column
- The conversion process will start
- A progress indicator will show the conversion is in progress

**For DOC/EXCEL files:**

- Click the refresh button (⟳) to start conversion
- Note: Conversion for these formats may take longer

### Step 4: View Markdown Content

1. Once conversion is complete, the status changes to "Converted"
2. Click the eye button (👁) in the Actions column
3. A modal dialog opens showing the markdown content
4. You can:
   - Scroll through the markdown
   - Download the markdown file using the download button (⬇)
   - Close the dialog by clicking the X button

### Step 5: Download Markdown

1. Open the markdown viewer (click the eye button)
2. Click the download button (⬇) in the top-right corner
3. The markdown file will be downloaded to your computer

## Features

### File Upload

- ✅ Drag and drop support
- ✅ Multiple file selection
- ✅ File type validation
- ✅ Progress tracking
- ✅ File size validation

### Document Management

- ✅ View all uploaded documents
- ✅ See conversion status
- ✅ File size and upload time display
- ✅ Pagination support
- ✅ Refresh functionality

### Markdown Conversion

- ✅ Automatic conversion for PDF/images
- ✅ Manual conversion for documents
- ✅ Progress indication
- ✅ Error handling with user-friendly messages

### Markdown Viewing

- ✅ Syntax-highlighted rendering
- ✅ Responsive layout
- ✅ Download functionality
- ✅ Maximizable dialog

## Troubleshooting

### Upload Fails

- Check file size (max 50MB)
- Verify file type is supported
- Ensure backend is running
- Check browser console for errors

### Conversion Fails

- Ensure vLLM service is running
- Check backend logs for errors
- Verify file is valid
- Try uploading a different file

### Cannot View Markdown

- Ensure conversion is complete (status shows "Converted")
- Refresh the page
- Check browser console for errors
- Verify backend is running

### Slow Conversion

- Conversion is time-consuming (may take several minutes)
- Larger files take longer
- Multiple pages in PDFs are processed in parallel
- Be patient and wait for completion

## API Endpoints

For developers, the following endpoints are available:

- `POST /upload` - Upload a file
- `GET /documents` - List all documents
- `POST /convert` - Convert a document
- `GET /markdown/{filename}` - Get markdown content

See `backend/DOCUMENT_API.md` for detailed API documentation.

## File Storage

### Input Directory

- Location: `backend/input/`
- Contains: Uploaded files
- Naming: Original filename preserved

### Output Directory

- Location: `backend/output/`
- Structure: `output/{filename}/{filename}_nohf.md`
- Contains: Converted markdown files

## Configuration

### Environment Variables

- `API_DOMAIN`: Backend API domain (default: http://localhost:8080)
- `API_PORT`: Backend port (default: 8080)
- `DOTS_OCR_VLLM_HOST`: Dots OCR vLLM server host (default: localhost)
- `DOTS_OCR_VLLM_PORT`: Dots OCR vLLM server port (default: 8001)
- `DOTS_OCR_VLLM_MODEL`: Dots OCR model name (default: dots_ocr)

### Frontend Configuration

- Edit `frontend/.env` to change API endpoint
- Example: `REACT_APP_CONFIG_API=http://localhost:8080/config`

## Tips and Best Practices

1. **Batch Upload**: Upload multiple files at once for efficiency
2. **Check Status**: Use the refresh button to check conversion status
3. **Download Results**: Download markdown files for backup
4. **File Organization**: Keep uploaded files organized by type
5. **Monitor Progress**: Watch the progress indicator during conversion

## Support

For issues or questions:

1. Check the browser console for error messages
2. Review backend logs for server errors
3. Verify all services are running
4. Check API connectivity
5. Review documentation in `backend/DOCUMENT_API.md`

## Next Steps

After converting documents to markdown:

1. Download the markdown files
2. Edit or process the markdown as needed
3. Integrate with your workflow
4. Archive original files if needed

Enjoy using the Document Upload and Conversion feature!
