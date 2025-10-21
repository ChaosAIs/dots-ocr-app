import * as React from "react";
import { useEffect, useState, useContext } from "react";

import { AuthContext } from "./authProvider";

export const Callback = () => {
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(true);
    const { signinRedirectCallback, clearAuthState, navigateToHome, debugAuthState } = useContext(AuthContext);

    useEffect(() => {
        const handleCallback = async () => {
            try {
                setLoading(true);
                // Debug current state
                if (debugAuthState) {
                    debugAuthState();
                }
                await signinRedirectCallback();
            } catch (err) {
                console.error("Authentication callback error:", err);
                setError(err.message);
                // Clear stale state and redirect to home after a delay
                clearAuthState();
                setTimeout(() => {
                    navigateToHome();
                }, 3000);
            } finally {
                setLoading(false);
            }
        };

        handleCallback();
    }, [signinRedirectCallback, clearAuthState, navigateToHome, debugAuthState]);

    if (loading) {
        return <div>Processing authentication...</div>;
    }

    // if (error) {
    //     return (
    //         <div>
    //             <p>Authentication error: {error}</p>
    //             <p>Redirecting to home page...</p>
    //         </div>
    //     );
    // }

    return <></>;
};