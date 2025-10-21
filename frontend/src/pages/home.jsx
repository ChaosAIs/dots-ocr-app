import { useState } from "react";
import { DocumentFileUpload } from "../components/documents/fileUpload";
import { DocumentList } from "../components/documents/documentList";
import "./home.scss";

export const Home = () => {
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const handleUploadSuccess = () => {
    // Trigger document list refresh
    setRefreshTrigger((prev) => prev + 1);
  };

  return (
    <div className="home-container">
      <div className="home-content">
        <DocumentFileUpload onUploadSuccess={handleUploadSuccess} />
        <DocumentList refreshTrigger={refreshTrigger} />
      </div>
    </div>
  );
};

