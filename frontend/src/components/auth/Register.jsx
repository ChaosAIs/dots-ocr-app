import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Card } from 'primereact/card';
import { InputText } from 'primereact/inputtext';
import { Password } from 'primereact/password';
import { Button } from 'primereact/button';
import { Toast } from 'primereact/toast';
import authService from '../../services/authService';

export const Register = () => {
    const navigate = useNavigate();
    const toast = React.useRef(null);

    const [formData, setFormData] = useState({
        username: '',
        email: '',
        fullName: '',
        password: '',
        confirmPassword: ''
    });

    const [loading, setLoading] = useState(false);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const validateForm = () => {
        if (!formData.username || !formData.email || !formData.password) {
            toast.current.show({
                severity: 'warn',
                summary: 'Validation Error',
                detail: 'Please fill in all required fields',
                life: 3000
            });
            return false;
        }

        if (formData.username.length < 3) {
            toast.current.show({
                severity: 'warn',
                summary: 'Validation Error',
                detail: 'Username must be at least 3 characters',
                life: 3000
            });
            return false;
        }

        if (formData.password.length < 8) {
            toast.current.show({
                severity: 'warn',
                summary: 'Validation Error',
                detail: 'Password must be at least 8 characters',
                life: 3000
            });
            return false;
        }

        if (formData.password !== formData.confirmPassword) {
            toast.current.show({
                severity: 'warn',
                summary: 'Validation Error',
                detail: 'Passwords do not match',
                life: 3000
            });
            return false;
        }

        // Email validation
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(formData.email)) {
            toast.current.show({
                severity: 'warn',
                summary: 'Validation Error',
                detail: 'Please enter a valid email address',
                life: 3000
            });
            return false;
        }

        return true;
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!validateForm()) {
            return;
        }

        setLoading(true);

        try {
            const result = await authService.register(
                formData.username,
                formData.email,
                formData.password,
                formData.fullName || null
            );

            if (result.success) {
                toast.current.show({
                    severity: 'success',
                    summary: 'Registration Successful',
                    detail: `Welcome, ${result.user.username}!`,
                    life: 2000
                });

                // Navigate to home after short delay
                setTimeout(() => {
                    navigate('/home');
                }, 1000);
            } else {
                toast.current.show({
                    severity: 'error',
                    summary: 'Registration Failed',
                    detail: result.error || 'Registration failed',
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

            <Card className="w-full md:w-30rem shadow-4">
                <div className="text-center mb-4">
                    <i className="pi pi-user-plus text-4xl text-primary mb-3" />
                    <h2 className="m-0 mb-2 text-color">Create Account</h2>
                    <p className="m-0 text-500">Sign up to get started</p>
                </div>

                <form onSubmit={handleSubmit}>
                    <div className="field mb-4">
                        <label htmlFor="username" className="block font-medium mb-2 text-color">Username *</label>
                        <InputText
                            id="username"
                            name="username"
                            value={formData.username}
                            onChange={handleChange}
                            placeholder="Choose a username"
                            className="w-full"
                            autoFocus
                        />
                    </div>

                    <div className="field mb-4">
                        <label htmlFor="email" className="block font-medium mb-2 text-color">Email *</label>
                        <InputText
                            id="email"
                            name="email"
                            type="email"
                            value={formData.email}
                            onChange={handleChange}
                            placeholder="Enter your email"
                            className="w-full"
                        />
                    </div>

                    <div className="field mb-4">
                        <label htmlFor="fullName" className="block font-medium mb-2 text-color">Full Name</label>
                        <InputText
                            id="fullName"
                            name="fullName"
                            value={formData.fullName}
                            onChange={handleChange}
                            placeholder="Enter your full name (optional)"
                            className="w-full"
                        />
                    </div>

                    <div className="field mb-4">
                        <label htmlFor="password" className="block font-medium mb-2 text-color">Password *</label>
                        <Password
                            id="password"
                            name="password"
                            value={formData.password}
                            onChange={handleChange}
                            placeholder="Create a password"
                            className="w-full"
                            toggleMask
                            feedback={false}
                        />
                        <small className="block mt-2 text-500 font-italic">
                            Must be at least 8 characters with uppercase, lowercase, and digit
                        </small>
                    </div>

                    <div className="field mb-4">
                        <label htmlFor="confirmPassword" className="block font-medium mb-2 text-color">Confirm Password *</label>
                        <Password
                            id="confirmPassword"
                            name="confirmPassword"
                            value={formData.confirmPassword}
                            onChange={handleChange}
                            placeholder="Confirm your password"
                            className="w-full"
                            feedback={false}
                            toggleMask
                        />
                    </div>

                    <Button
                        type="submit"
                        label="Create Account"
                        icon="pi pi-user-plus"
                        className="w-full"
                        loading={loading}
                    />

                    <div className="text-center mt-4">
                        <p className="m-0 text-500">
                            Already have an account?{' '}
                            <Link to="/login" className="text-primary font-semibold no-underline hover:underline">
                                Sign in
                            </Link>
                        </p>
                    </div>
                </form>
            </Card>
        </div>
    );
};

