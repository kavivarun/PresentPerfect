import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import UploadPage from './pages/UploadPage';
import ReportPage from './pages/ReportPage';
import Header from './components/Header';
import { AuthProvider } from './context/AuthContext';

export default function App() {
  return (
    <>
    <AuthProvider>
    <Header />
    <main>
      <Routes>
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/faqs" element={<UploadPage />} />
        <Route path="/about" element={<UploadPage />} />
        <Route path="/report" element={<ReportPage />} />
        <Route path="*" element={<Navigate to="/upload" replace />} />
      </Routes>
    </main>
    </AuthProvider>
    </>
  );
}