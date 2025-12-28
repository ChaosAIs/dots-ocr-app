import { useCallback } from "react";
import { DocumentList } from "../components/documents/documentList";
import { WorkspaceSidebar } from "../components/workspace/WorkspaceSidebar";
import { useWorkspace } from "../contexts/WorkspaceContext";

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
    <div className="flex" style={{ height: 'calc(100vh - 60px)', backgroundColor: 'var(--surface-ground)' }}>
      {/* Sidebar */}
      <div
        className="flex-shrink-0 flex flex-column"
        style={{
          width: '280px',
          backgroundColor: 'var(--surface-card)',
          borderRight: '1px solid var(--surface-border)'
        }}
      >
        <WorkspaceSidebar
          selectedWorkspace={currentWorkspace}
          onWorkspaceSelect={handleWorkspaceSelect}
          onWorkspaceChange={handleWorkspaceChange}
        />
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto" style={{ backgroundColor: 'var(--surface-card)' }}>
        {/* Only show DocumentList when there are workspaces and one is selected */}
        {!loading && showDocumentList && <DocumentList />}

        {/* Show message when no workspaces exist */}
        {!loading && !hasWorkspaces && (
          <div className="flex flex-column align-items-center justify-content-center" style={{ minHeight: '300px' }}>
            <i className="pi pi-folder-open text-4xl mb-3" style={{ color: 'var(--text-color-secondary)' }} />
            <p className="m-0" style={{ color: 'var(--text-color-secondary)' }}>No workspaces yet. Create a workspace to start uploading documents.</p>
          </div>
        )}
      </div>
    </div>
  );
};
