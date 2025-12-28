import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Card } from 'primereact/card';
import { InputText } from 'primereact/inputtext';
import { Password } from 'primereact/password';
import { Button } from 'primereact/button';
import { Checkbox } from 'primereact/checkbox';
import { Toast } from 'primereact/toast';
import { useAuth } from '../../core/auth/components/authProvider';

export const Login = () => {
    const navigate = useNavigate();
    const { login } = useAuth();
    const toast = React.useRef(null);

    const [formData, setFormData] = useState({
        username: '',
        password: '',
        rememberMe: false
    });

    const [loading, setLoading] = useState(false);

    const handleChange = (e) => {
        const { name, value, checked, type } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: type === 'checkbox' ? checked : value
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!formData.username || !formData.password) {
            toast.current.show({
                severity: 'warn',
                summary: 'Validation Error',
                detail: 'Please enter username and password',
                life: 3000
            });
            return;
        }

        setLoading(true);

        try {
            // Use the context's login function to update React state properly
            const result = await login(formData.username, formData.password);

            if (result.success) {
                toast.current.show({
                    severity: 'success',
                    summary: 'Login Successful',
                    detail: `Welcome back, ${result.user.username}!`,
                    life: 2000
                });

                // Navigate to home after short delay
                setTimeout(() => {
                    navigate('/home');
                }, 1000);
            } else {
                toast.current.show({
                    severity: 'error',
                    summary: 'Login Failed',
                    detail: result.error || 'Invalid credentials',
                    life: 4000
                });
            }
        } catch (error) {
            toast.current.show({
                severity: 'error',
                summary: 'Error',
                detail: 'An unexpected error occurred',
                life: 4000
            });
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex align-items-center justify-content-center min-h-screen surface-ground p-4">
            <Toast ref={toast} />

            <Card className="w-full md:w-25rem shadow-4">
                <div className="text-center mb-4">
                    <i className="pi pi-sign-in text-4xl text-primary mb-3" />
                    <h2 className="m-0 mb-2 text-color">Welcome Back</h2>
                    <p className="m-0 text-500">Sign in to your account</p>
                </div>

                <form onSubmit={handleSubmit}>
                    <div className="field mb-4">
                        <label htmlFor="username" className="block font-medium mb-2 text-color">Username or Email</label>
                        <InputText
                            id="username"
                            name="username"
                            value={formData.username}
                            onChange={handleChange}
                            placeholder="Enter your username or email"
                            className="w-full"
                            autoFocus
                        />
                    </div>

                    <div className="field mb-4">
                        <label htmlFor="password" className="block font-medium mb-2 text-color">Password</label>
                        <Password
                            id="password"
                            name="password"
                            value={formData.password}
                            onChange={handleChange}
                            placeholder="Enter your password"
                            className="w-full"
                            feedback={false}
                            toggleMask
                        />
                    </div>

                    <div className="flex align-items-center mb-4">
                        <Checkbox
                            inputId="rememberMe"
                            name="rememberMe"
                            checked={formData.rememberMe}
                            onChange={handleChange}
                        />
                        <label htmlFor="rememberMe" className="ml-2 text-500">Remember me</label>
                    </div>

                    <Button
                        type="submit"
                        label="Sign In"
                        icon="pi pi-sign-in"
                        className="w-full"
                        loading={loading}
                    />

                    <div className="text-center mt-4">
                        <p className="m-0 text-500">
                            Don't have an account?{' '}
                            <Link to="/register" className="text-primary font-semibold no-underline hover:underline">
                                Sign up
                            </Link>
                        </p>
                    </div>
                </form>
            </Card>
        </div>
    );
};

