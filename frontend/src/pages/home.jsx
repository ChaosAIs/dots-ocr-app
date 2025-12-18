import { useState, useRef } from "react";
import { DocumentFileUpload } from "../components/documents/fileUpload";
import { DocumentList } from "../components/documents/documentList";
import "./home.scss";

export const Home = () => {
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const documentListRef = useRef(null);

  const handleUploadSuccess = (uploadedDocIds) => {
    console.log(`🏠 Home: Upload success with ${uploadedDocIds?.length || 0} document IDs:`, uploadedDocIds);

    // Subscribe to newly uploaded documents via DocumentList
    if (documentListRef.current && uploadedDocIds && uploadedDocIds.length > 0) {
      documentListRef.current.subscribeToNewDocuments(uploadedDocIds);
    }

    // Trigger document list refresh
    setRefreshTrigger((prev) => prev + 1);
  };

  return (
    <div className="home-container">
      <div className="home-content">
        <DocumentFileUpload onUploadSuccess={handleUploadSuccess} />
        <DocumentList ref={documentListRef} refreshTrigger={refreshTrigger} />
      </div>
    </div>
  );
};

