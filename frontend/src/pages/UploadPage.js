import React, { useState } from 'react';
import DocumentList from '../components/DocumentList';
import axios from 'axios';

const UploadPage = () => {
  const [file, setFile] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);

  // Загрузка списка документов
  const fetchDocuments = async () => {
    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      const res = await axios.get('/documents', { headers: { Authorization: `Bearer ${token}` } });
      setDocuments(res.data);
    } catch (e) {
      setDocuments([]);
    }
    setLoading(false);
  };

  React.useEffect(() => {
    fetchDocuments();
  }, []);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return alert('Выберите файл для загрузки!');
    setUploading(true);
    try {
      const token = localStorage.getItem('token');
      const formData = new FormData();
      formData.append('file', file);
      formData.append('title', file.name);
      formData.append('source', 'manual');
      await axios.post('/documents/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data', Authorization: `Bearer ${token}` },
      });
      setFile(null);
      await fetchDocuments();
      alert('Файл успешно загружен и проиндексирован!');
    } catch (err) {
      alert('Ошибка загрузки: ' + (err?.response?.data?.detail || err.message));
    }
    setUploading(false);
  };

  const handleDelete = async (docId) => {
    if (!window.confirm('Удалить документ?')) return;
    try {
      const token = localStorage.getItem('token');
      await axios.delete(`/documents/${docId}`, { headers: { Authorization: `Bearer ${token}` } });
      await fetchDocuments();
    } catch (err) {
      alert('Ошибка удаления: ' + (err?.response?.data?.detail || err.message));
    }
  };

  const handleClearIndex = async () => {
    if (!window.confirm('Очистить весь векторный индекс?')) return;
    try {
      const token = localStorage.getItem('token');
      await axios.post('/index/clear', {}, { headers: { Authorization: `Bearer ${token}` } });
      alert('Индекс очищен!');
    } catch (err) {
      alert('Ошибка очистки индекса: ' + (err?.response?.data?.detail || err.message));
    }
  };

  return (
    <div style={{ padding: 32 }}>
      <h2>Загрузка документов</h2>
      <form onSubmit={handleUpload} style={{ marginBottom: 24 }}>
        <input type="file" onChange={handleFileChange} />
        <button type="submit" style={{ marginLeft: 16 }} disabled={uploading}>{uploading ? 'Загрузка...' : 'Загрузить'}</button>
        {file && <span style={{ marginLeft: 16 }}>Выбран файл: {file.name}</span>}
      </form>
      <button onClick={handleClearIndex} style={{ marginBottom: 16, background: '#e53935', color: 'white', border: 'none', padding: '8px 16px', borderRadius: 4, cursor: 'pointer' }}>
        Очистить индекс
      </button>
      <DocumentList documents={documents} loading={loading} onDelete={handleDelete} />
    </div>
  );
};

export default UploadPage;