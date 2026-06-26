import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { setToken } from '../api/client'
import { KeyRound, Loader2, Zap } from 'lucide-react'
import axios from 'axios'

export default function Login() {
  const [token, setTokenInput] = useState('')
  const [error, setError] = useState('')
  const [fetching, setFetching] = useState(false)
  const navigate = useNavigate()

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const t = token.trim()
    if (!t) {
      setError('Paste a JWT token or click "Get Dev Token"')
      return
    }
    setToken(t)
    navigate('/chat')
  }

  const fetchDevToken = async () => {
    setFetching(true)
    setError('')
    try {
      const { data } = await axios.post('/api/auth/token?tenant_id=acme&user_id=user1&role=admin')
      setTokenInput(data.token)
    } catch {
      setError('Could not reach the server. Make sure FastAPI is running on port 8000.')
    } finally {
      setFetching(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="w-12 h-12 bg-indigo-600 rounded-2xl flex items-center justify-center mx-auto mb-4">
            <KeyRound size={22} />
          </div>
          <h1 className="text-xl font-semibold text-white">Enterprise Doc AI</h1>
          <p className="text-gray-500 text-sm mt-1">Sign in to continue</p>
        </div>

        <div className="bg-gray-900 rounded-2xl p-6 border border-gray-800 space-y-4">

          {/* Quick dev login */}
          <button
            type="button"
            onClick={fetchDevToken}
            disabled={fetching}
            className="w-full flex items-center justify-center gap-2 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 rounded-lg text-sm font-medium transition-colors"
          >
            {fetching
              ? <><Loader2 size={15} className="animate-spin" /> Getting token…</>
              : <><Zap size={15} /> Get Dev Token</>
            }
          </button>

          <div className="flex items-center gap-3">
            <div className="flex-1 h-px bg-gray-800" />
            <span className="text-xs text-gray-600">or paste manually</span>
            <div className="flex-1 h-px bg-gray-800" />
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-gray-400 mb-1.5">JWT Token</label>
              <textarea
                value={token}
                onChange={(e) => { setTokenInput(e.target.value); setError('') }}
                placeholder="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
                rows={3}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-xs text-gray-300 font-mono placeholder-gray-600 focus:outline-none focus:border-indigo-500 resize-none"
              />
              {error && <p className="text-red-400 text-xs mt-1">{error}</p>}
            </div>

            <button
              type="submit"
              disabled={!token.trim()}
              className="w-full py-2.5 bg-gray-700 hover:bg-gray-600 disabled:opacity-40 disabled:cursor-not-allowed rounded-lg text-sm font-medium transition-colors"
            >
              Continue with pasted token
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}
