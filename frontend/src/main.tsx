import React, { lazy, Suspense } from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { ProtectedRoute } from './components/ProtectedRoute'
import { ProtectedAdminRoute } from './components/ProtectedAdminRoute'
import { AuthInitializer } from './components/AuthInitializer'
import { ErrorBoundary } from './components/ErrorBoundary'
import './styles/globals.css'

const App = lazy(() => import('./App'))
const Login = lazy(() => import('./pages/Login').then(module => ({ default: module.Login })))
const Register = lazy(() => import('./pages/Register').then(module => ({ default: module.Register })))
const Setup = lazy(() => import('./pages/Setup').then(module => ({ default: module.Setup })))
const ApiKeys = lazy(() => import('./pages/ApiKeys').then(module => ({ default: module.ApiKeys })))
const Admin = lazy(() => import('./pages/Admin').then(module => ({ default: module.Admin })))

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <BrowserRouter>
      <AuthInitializer>
        <Suspense fallback={<div className="min-h-screen bg-surface-900" />}>
        <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/setup" element={<Setup />} />
        <Route path="/api-keys" element={<ProtectedRoute><ApiKeys /></ProtectedRoute>} />
        <Route path="/admin" element={<ProtectedAdminRoute><Admin /></ProtectedAdminRoute>} />
        <Route path="/*" element={<ProtectedRoute><App /></ProtectedRoute>} />
        </Routes>
        </Suspense>
      </AuthInitializer>
      </BrowserRouter>
    </ErrorBoundary>
  </React.StrictMode>,
)
