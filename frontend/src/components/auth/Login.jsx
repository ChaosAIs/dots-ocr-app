import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Card } from 'primereact/card';
import { InputText } from 'primereact/inputtext';
import { Password } from 'primereact/password';
import { Button } from 'primereact/button';
import { Checkbox } from 'primereact/checkbox';
import { Toast } from 'primereact/toast';
import authService from '../../services/authService';
import './Login.scss';

export const Login = () => {
    const navigate = useNavigate();
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
            const result = await authService.login(formData.username, formData.password);

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
        <div className="login-container">
            <Toast ref={toast} />
            
            <Card className="login-card">
                <div className="login-header">
                    <i className="pi pi-sign-in" style={{ fontSize: '3rem', color: '#667eea' }}></i>
                    <h2>Welcome Back</h2>
                    <p>Sign in to your account</p>
                </div>

                <form onSubmit={handleSubmit} className="login-form">
                    <div className="p-field">
                        <label htmlFor="username">Username or Email</label>
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

                    <div className="p-field">
                        <label htmlFor="password">Password</label>
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

                    <div className="p-field-checkbox">
                        <Checkbox
                            inputId="rememberMe"
                            name="rememberMe"
                            checked={formData.rememberMe}
                            onChange={handleChange}
                        />
                        <label htmlFor="rememberMe">Remember me</label>
                    </div>

                    <Button
                        type="submit"
                        label="Sign In"
                        icon="pi pi-sign-in"
                        className="w-full"
                        loading={loading}
                    />

                    <div className="login-footer">
                        <p>
                            Don't have an account?{' '}
                            <Link to="/register" className="register-link">
                                Sign up
                            </Link>
                        </p>
                    </div>
                </form>
            </Card>
        </div>
    );
};

