import { MessageSquare, FileText, LogOut } from 'lucide-react'
import { NavLink } from 'react-router-dom'
import { clearToken } from '../../api/client'

export default function Sidebar() {
  const handleLogout = () => {
    clearToken()
    window.location.href = '/login'
  }

  return (
    <aside className="w-56 bg-gray-900 border-r border-gray-800 flex flex-col h-screen fixed left-0 top-0">
      <div className="p-4 border-b border-gray-800">
        <h1 className="text-white font-semibold text-sm">Enterprise Doc AI</h1>
        <p className="text-gray-500 text-xs mt-0.5">RAG Pipeline</p>
      </div>

      <nav className="flex-1 p-3 space-y-1">
        <NavLink
          to="/chat"
          className={({ isActive }) =>
            `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
              isActive
                ? 'bg-indigo-600 text-white'
                : 'text-gray-400 hover:bg-gray-800 hover:text-white'
            }`
          }
        >
          <MessageSquare size={16} />
          Chat
        </NavLink>

        <NavLink
          to="/documents"
          className={({ isActive }) =>
            `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
              isActive
                ? 'bg-indigo-600 text-white'
                : 'text-gray-400 hover:bg-gray-800 hover:text-white'
            }`
          }
        >
          <FileText size={16} />
          Documents
        </NavLink>
      </nav>

      <div className="p-3 border-t border-gray-800">
        <button
          onClick={handleLogout}
          className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-gray-400 hover:bg-gray-800 hover:text-white transition-colors w-full"
        >
          <LogOut size={16} />
          Logout
        </button>
      </div>
    </aside>
  )
}
