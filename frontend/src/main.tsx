import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import App from './App'
import { Login } from './pages/Login'
import { Register } from './pages/Register'
import { Setup } from './pages/Setup'
import { ApiKeys } from './pages/ApiKeys'
import { Admin } from './pages/Admin'
import { ProtectedRoute } from './components/ProtectedRoute'
import { ProtectedAdminRoute } from './components/ProtectedAdminRoute'
import './styles/globals.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/setup" element={<Setup />} />
        <Route path="/api-keys" element={<ProtectedRoute><ApiKeys /></ProtectedRoute>} />
        <Route path="/admin" element={<ProtectedAdminRoute><Admin /></ProtectedAdminRoute>} />
        <Route path="/*" element={<ProtectedRoute><App /></ProtectedRoute>} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
)
