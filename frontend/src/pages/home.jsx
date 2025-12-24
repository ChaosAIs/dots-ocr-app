import { useCallback } from "react";
import { DocumentList } from "../components/documents/documentList";
import { WorkspaceSidebar } from "../components/workspace/WorkspaceSidebar";
import { useWorkspace } from "../contexts/WorkspaceContext";
import "./home.scss";

export const Home = () => {
  // Use centralized workspace context
  const {
    workspaces,
    currentWorkspace,
    selectWorkspace,
    loadWorkspaces,
    loading,
  } = useWorkspace();

  const handleWorkspaceSelect = useCallback((workspace) => {
    console.log("🏠 Home: Workspace selected:", workspace?.name, "ID:", workspace?.id);
    selectWorkspace(workspace);
  }, [selectWorkspace]);

  const handleWorkspaceChange = useCallback(() => {
    // Reload workspaces when one is created/updated/deleted
    console.log("🏠 Home: Workspace changed, reloading...");
    loadWorkspaces();
  }, [loadWorkspaces]);

  // Determine if we should show the document list:
  // - Only show if there are workspaces AND a workspace is selected
  const hasWorkspaces = workspaces && workspaces.length > 0;
  const hasSelectedWorkspace = currentWorkspace !== null;
  const showDocumentList = hasWorkspaces && hasSelectedWorkspace;

  return (
    <div className="home-container">
      <WorkspaceSidebar
        selectedWorkspace={currentWorkspace}
        onWorkspaceSelect={handleWorkspaceSelect}
        onWorkspaceChange={handleWorkspaceChange}
      />
      <div className="home-content">
        {/* Only show DocumentList when there are workspaces and one is selected */}
        {!loading && showDocumentList && <DocumentList />}

        {/* Show message when no workspaces exist */}
        {!loading && !hasWorkspaces && (
          <div className="no-workspace-message">
            <i className="pi pi-folder-open" style={{ fontSize: '3rem', color: '#ccc', marginBottom: '1rem' }} />
            <p>No workspaces yet. Create a workspace to start uploading documents.</p>
          </div>
        )}
      </div>
    </div>
  );
};
