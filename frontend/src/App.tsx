import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Sidebar from './components/Layout/Sidebar'
import ChatPage from './components/Chat/ChatPage'
import DocumentsPage from './components/Documents/DocumentsPage'
import Login from './components/Login'
import { getToken } from './api/client'

const qc = new QueryClient()

function ProtectedLayout() {
  if (!getToken()) return <Navigate to="/login" replace />
  return (
    <div className="flex">
      <Sidebar />
      <main className="ml-56 flex-1 min-h-screen bg-gray-950">
        <Routes>
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/documents" element={<DocumentsPage />} />
          <Route path="*" element={<Navigate to="/chat" replace />} />
        </Routes>
      </main>
    </div>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/*" element={<ProtectedLayout />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
