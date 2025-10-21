import React, { useState, useEffect, useRef } from "react";
import { ProgressSpinner } from "primereact/progressspinner";
import { BlockUI } from "primereact/blockui";
import "./appProgress.scss";
import { loadingService } from "../loadingService";

const AppProgress = () => {
  const [isLoading, setIsLoading] = useState(false);
  const subscriptionRef = useRef(null);
  const timeoutRef = useRef(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    // Subscribe to loading service
    subscriptionRef.current = loadingService.get().subscribe((data) => {
      // Clear any pending timeout
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }

      // Add a small delay to prevent rapid state changes that can cause BlockUI issues
      timeoutRef.current = setTimeout(() => {
        if (mountedRef.current) {
          setIsLoading(data);
        }
      }, 20000); // Increased delay to better handle rapid changes
    });

    // Cleanup function
    return () => {
      mountedRef.current = false;

      // Clear any pending timeout
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }

      // Unsubscribe to ensure no memory leaks
      if (subscriptionRef.current) {
        subscriptionRef.current.unsubscribe();
      }
    };
  }, []);

  return (
    <>
      {/* Use a key to force re-render when loading state changes to prevent BlockUI issues */}
      <BlockUI
        key={`blockui-${isLoading}`}
        blocked={isLoading}
        fullScreen
      />
      {isLoading && (
        <div className="progress-spinner-center">
          <ProgressSpinner />
        </div>
      )}
    </>
  );
};

export default AppProgress;
