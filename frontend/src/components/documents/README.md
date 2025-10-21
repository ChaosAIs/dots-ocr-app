# Document Upload and Conversion Components

This directory contains components for uploading documents and converting them to markdown format.

## Components

### DocumentFileUpload
File upload component that allows users to upload PDF, images, and office documents.

**Features:**
- Drag and drop support
- Multiple file upload
- File type validation
- Progress tracking
- File size limit (50MB)

**Props:**
- `onUploadSuccess`: Callback function triggered after successful upload

**Usage:**
```jsx
import { DocumentFileUpload } from './fileUpload';

<DocumentFileUpload onUploadSuccess={handleUploadSuccess} />
```

### DocumentList
Displays a list of uploaded documents with their conversion status.

**Features:**
- DataTable with pagination
- View markdown button for converted documents
- Convert button for pending documents
- File size and upload time display
- Refresh functionality

**Props:**
- `refreshTrigger`: Trigger value to refresh the document list

**Usage:**
```jsx
import { DocumentList } from './documentList';

<DocumentList refreshTrigger={refreshTrigger} />
```

### MarkdownViewer
Modal dialog for viewing markdown content of converted documents.

**Features:**
- Syntax-highlighted markdown rendering
- Download markdown file
- Maximizable dialog
- Responsive layout

**Props:**
- `document`: Document object with filename
- `visible`: Boolean to control dialog visibility
- `onHide`: Callback when dialog is closed

**Usage:**
```jsx
import MarkdownViewer from './markdownViewer';

<MarkdownViewer 
  document={document}
  visible={showViewer}
  onHide={handleHide}
/>
```

## Services

### DocumentService
Service for handling all document-related API calls.

**Methods:**
- `uploadDocument(file)`: Upload a file
- `getDocuments()`: Get list of all documents
- `convertDocument(filename, promptMode)`: Convert document to markdown
- `getMarkdownContent(filename)`: Get markdown content
- `formatFileSize(bytes)`: Format file size for display
- `formatDate(isoString)`: Format date for display

**Usage:**
```jsx
import documentService from '../../services/documentService';

const response = await documentService.uploadDocument(file);
const documents = await documentService.getDocuments();
```

## Styling

Each component has its own SCSS file for styling:
- `fileUpload.scss`: File upload component styles
- `documentList.scss`: Document list component styles
- `markdownViewer.scss`: Markdown viewer component styles

## API Endpoints

The components interact with the following backend endpoints:

- `POST /upload`: Upload a document
- `GET /documents`: Get list of documents
- `POST /convert`: Convert document to markdown
- `GET /markdown/{filename}`: Get markdown content

## Supported File Types

- PDF (.pdf)
- Images (.png, .jpg, .jpeg, .gif, .bmp)
- Documents (.doc, .docx)
- Spreadsheets (.xls, .xlsx)

## Dependencies

- `primereact`: UI components
- `react-markdown`: Markdown rendering
- `axios`: HTTP client

## Integration

To integrate these components into your application:

1. Import the components in your page/container
2. Set up state management for refresh triggers
3. Ensure the backend API is running and accessible
4. Configure the API domain in `APP_CONFIG`

Example:
```jsx
import { DocumentFileUpload } from '../components/documents/fileUpload';
import { DocumentList } from '../components/documents/documentList';

export const Home = () => {
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const handleUploadSuccess = () => {
    setRefreshTrigger(prev => prev + 1);
  };

  return (
    <div>
      <DocumentFileUpload onUploadSuccess={handleUploadSuccess} />
      <DocumentList refreshTrigger={refreshTrigger} />
    </div>
  );
};
```

