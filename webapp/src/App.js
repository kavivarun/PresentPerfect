import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import UploadPage from './pages/UploadPage';
import Header from './components/Header';

export default function App() {
  return (
    <>
    <Header />
    <main>
      <Routes>
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/faqs" element={<UploadPage />} />
        <Route path="/about" element={<UploadPage />} />
        <Route path="*" element={<Navigate to="/upload" replace />} />
      </Routes>
    </main>
    </>
  );
}