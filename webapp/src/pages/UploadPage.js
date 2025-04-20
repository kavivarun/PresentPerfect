import React, { useState, useCallback, useEffect, useRef, } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import LoadingBar from '../components/LoadingBar';
import { ReactComponent as UploadIcon } from '../assets/upload.svg';
import { ReactComponent as CheckBox } from '../assets/checkbox.svg';
import { ReactComponent as UploadFile } from '../assets/uploadfile.svg';
import { ReactComponent as DropVideo } from '../assets/videodrop.svg';
import { io } from 'socket.io-client';

export default function UploadPage() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [toast, setToast] = useState(null);
  const [toastVisible, setToastVisible] = useState(false);
  const socketRef = useRef(null);
  const [processingMsg, setProcessingMsg] = useState('');

  useEffect(() => {
    socketRef.current = io('http://localhost:4000'); 

    socketRef.current.on('processing-update', (data) => {
      console.log('Progress:', data);
      setProcessingMsg(data.message);
    });

    socketRef.current.on('processing-complete', (data) => {
      console.log('Done:', data.message);
      setProcessingMsg(data.message);
    });

    return () => {
      socketRef.current.disconnect();
    };
  }, []);

  const showToast = (message) => {
    setToast(message);
    setToastVisible(true);
    setTimeout(() => {
      setToastVisible(false);
      setTimeout(() => setToast(null), 400); // matches exit animation
    }, 3000);
  };

  const benefits = [
    "Real-time posture analysis to correct body language issues like slouching",
    "Confidence feedback by detecting emotion, movement and stance",
    "Practice smarter — get actionable insights instantly for real presentations"
  ];

  const onDrop = useCallback(
    async (acceptedFiles, fileRejections) => {
      if (fileRejections.length > 0) {
        showToast("Please upload a valid .mp4 video file.");
        return;
      }
      if (!acceptedFiles.length) return;

      const file = acceptedFiles[0];
      const formData = new FormData();
      formData.append('video', file);

      setLoading(true);
      try {
        const resp = await axios.post('http://localhost:4000/api/analyze', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          onUploadProgress: (evt) => {
            const pct = Math.round((evt.loaded / evt.total) * 100);
            setProgress(pct);
          }
        });
        //navigate('/report', { state: resp.data });
      } catch (err) {
        console.error(err);
        alert('Upload failed');
      } finally {
        setLoading(false);
        setProgress(0);
      }
    },
    [navigate]
  );

  const {
    getRootProps,
    getInputProps,
    isDragActive,
    open
  } = useDropzone({
    onDrop,
    accept: { 'video/mp4': [] },
    multiple: false,
    noClick: true,
    noKeyboard: true,
  });

  const styles = {
    page: {
      padding: '1rem',
      backgroundColor: '#f9f9ff',
      position: 'relative',
      zIndex: 1,
    },
    heading: {
      textAlign: 'center',
      fontSize: '2.5rem',
      marginBottom: '.5rem',
    },
    uploadBox: {
      boxSizing: 'border-box',
      width: '90%',
      minHeight: '300px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      backgroundColor: '#5D2E8C',
      borderRadius: '1.5rem',
      margin: '0 auto 1rem',
      position: 'relative',
      boxShadow: '0 4px 10px rgba(0,0,0,0.3)',
      outline: '2px dashed rgba(255,255,255,0.0)',
      outlineOffset: '-2px',
      zIndex: 2,
      transition: 'outline-color 0.2s ease',
    },
    innerBorder: {
      position: 'absolute',
      top: '12px',
      bottom: '12px',
      left: '12px',
      right: '12px',
      border: '2px dashed rgba(255,255,255,0.6)',
      borderRadius: '1.25rem',
      pointerEvents: 'none',
      transition: 'border-color 0.2s ease, border-style 0.2s ease',
    },
    dragOverlay: {
      position: 'absolute',
      top: 0,
      left: 0,
      width: '100%',
      height: '100%',
      backgroundColor: 'rgba(0, 0, 0, 0.6)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      borderRadius: '1.5rem',
      zIndex: 3,
      pointerEvents: 'none',
      color: 'white',
      textAlign: 'center',
    },
    uploadContent: {
      position: 'relative',
      zIndex: 2,
      textAlign: 'center',
      justifyItems: 'center',
      color: 'white',
    },
    icon: {
      fill: '#f9f9ff',
    },
    uploadButton: {
      marginTop: '1rem',
      color: 'black',
      fontSize: '1.1rem',
      padding: '0.8rem 2rem',
      borderRadius: '1rem',
      border: 'none',
      cursor: 'pointer',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '0.5rem',
      transition: 'all 0.5s ease-in-out',
    },
    section: {
      display: 'flex',
      gap: '2rem',
      margin: '0 auto 3rem',
      marginLeft: '5%',
      position: 'relative',
      zIndex: 2,
    },
    instructions: {
      flex: 1,
      fontSize: '1rem',
      lineHeight: '1.6',
    },
    checklist: {
      flex: 1,
      fontSize: '1rem',
    },
    checkItem: {
      display: 'flex',
      alignItems: 'center',
      gap: '0.5rem',
      marginBottom: '1rem',
    },
    checkIcon: {
      color: '#5e2b97',
      fontSize: '1.2rem',
    },
    triangleBg: {
      position: 'fixed',
      top: '40%',
      right: 0,
      width: '100%',
      height: '60%',
      opacity: '10%',
      backgroundColor: '#B288C0',
      zIndex: 0,
      clipPath: 'polygon(70% 15%, 0% 100%, 85% 100%)',
      pointerEvents: 'none',
    },
    toast: {
      position: 'fixed',
      bottom: '2rem',
      left: '50%',
      transform: 'translateX(-50%)',
      backgroundColor: '#5D2E8C',
      color: 'white',
      padding: '1rem 2rem',
      borderRadius: '1rem',
      boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
      zIndex: 999,
      fontSize: '1rem',
      animation: `${toastVisible ? 'fadeSlideIn' : 'fadeSlideOut'} 0.4s forwards`,
      pointerEvents: 'none',
    },
  };

  return (
    <div style={styles.page}>
      <h1 style={styles.heading}>Analyse Yourself</h1>

      <div {...getRootProps()} style={styles.uploadBox}>
        <input {...getInputProps()} />
        <div style={styles.innerBorder} />

        {isDragActive && (
          <div style={styles.dragOverlay}>
            <DropVideo style={{ width: '5rem', height: '5rem', stroke: '#fff' }} />
            <p style={{ marginTop: '1rem', fontSize: '1.2rem' }}>Drop it like it's HOT</p>
          </div>
        )}

        <div style={styles.uploadContent}>
          {!isDragActive && (
            <>
              <div style={styles.icon}>
                <UploadIcon style={{ width: '8rem', height: '8rem', stroke: '#f9f9ff' }} />
              </div>
              <div>Drag and drop your video to upload</div>
              <button type="button" onClick={open} style={styles.uploadButton}>
                <UploadFile style={{ width: '1.5rem', height: '1.5rem', stroke: '#000' }} />
                Choose Video
              </button>
            </>
          )}
        </div>
      </div>


      <div style={styles.section}>
        <div style={styles.instructions}>
          <h2>Instructions</h2>
          <p>
            Nervous about speaking in front of a crowd? Don’t worry — we’ve got your back.
            Simply upload a short video of yourself speaking. Our AI will analyze your posture,
            gestures, and movement patterns to provide insightful feedback.
            <br />
            &nbsp; &nbsp; &nbsp;Make sure your entire body is clearly visible<br />
            &nbsp; &nbsp; &nbsp;Record in a well-lit environment with minimal background clutter<br />
            Once submitted, your results will appear with suggestions to help improve your delivery.
          </p>
        </div>

        <div style={styles.checklist}>
          <h2>Why use PresentPerfect?</h2>
          {benefits.map((text, i) => (
            <div key={i} style={styles.checkItem}>
              <span style={styles.checkIcon}>
                <CheckBox style={{ width: '1.5rem', height: '1.5rem', fill: '#5D2E8C' }} />
              </span>
              <span>{text}</span>
            </div>
          ))}
        </div>
      </div>

      <div style={styles.triangleBg} />

      {toast && <div style={styles.toast}>{toast}</div>}
    </div>
  );
}
