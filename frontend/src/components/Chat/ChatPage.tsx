import { useState, useRef, useEffect } from 'react'
import { Send, Bot, User, ChevronDown, ChevronUp, Loader2 } from 'lucide-react'
import { queryDocuments } from '../../api/query'
import type { ChatMessage, ChunkResult } from '../../types/api'

function SourceChunks({ chunks }: { chunks: ChunkResult[] }) {
  const [open, setOpen] = useState(false)
  if (!chunks.length) return null
  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 text-xs text-indigo-400 hover:text-indigo-300"
      >
        {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        {chunks.length} source{chunks.length > 1 ? 's' : ''}
      </button>
      {open && (
        <div className="mt-2 space-y-2">
          {chunks.map((c, i) => (
            <div key={c.chunk_id} className="bg-gray-900 rounded p-2 border border-gray-700">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-gray-500 font-mono">
                  Source {i + 1} · score {c.score.toFixed(3)}
                </span>
                <span className="text-xs text-gray-600 truncate max-w-48">
                  {(c.metadata?.file_name as string) || c.document_id.slice(0, 8)}
                </span>
              </div>
              <p className="text-xs text-gray-400 leading-relaxed line-clamp-3">{c.text}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function MessageBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === 'user'
  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div
        className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 ${
          isUser ? 'bg-indigo-600' : 'bg-gray-700'
        }`}
      >
        {isUser ? <User size={14} /> : <Bot size={14} />}
      </div>
      <div className={`max-w-2xl ${isUser ? 'items-end' : 'items-start'} flex flex-col`}>
        <div
          className={`px-4 py-2.5 rounded-2xl text-sm leading-relaxed ${
            isUser
              ? 'bg-indigo-600 text-white rounded-tr-sm'
              : 'bg-gray-800 text-gray-200 rounded-tl-sm'
          }`}
        >
          {msg.content}
        </div>
        {msg.chunks && <SourceChunks chunks={msg.chunks} />}
        {msg.retrieval_stats && (
          <p className="text-xs text-gray-600 mt-1">
            {msg.retrieval_stats.n_after_rerank ?? 0} chunks retrieved
          </p>
        )}
      </div>
    </div>
  )
}

function TypingIndicator() {
  return (
    <div className="flex gap-3">
      <div className="w-7 h-7 rounded-full bg-gray-700 flex items-center justify-center">
        <Bot size={14} />
      </div>
      <div className="bg-gray-800 rounded-2xl rounded-tl-sm px-4 py-3">
        <Loader2 size={14} className="animate-spin text-gray-400" />
      </div>
    </div>
  )
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '0',
      role: 'assistant',
      content: 'Hi! Ask me anything about your ingested documents.',
      timestamp: new Date(),
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const send = async () => {
    const text = input.trim()
    if (!text || loading) return
    setInput('')

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: text,
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, userMsg])
    setLoading(true)

    try {
      const res = await queryDocuments(text, 5, true)
      const aiMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: res.answer || 'No answer generated.',
        chunks: res.chunks,
        retrieval_stats: res.retrieval_stats,
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, aiMsg])
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: 'Something went wrong. Check the server is running.',
          timestamp: new Date(),
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-screen">
      <div className="border-b border-gray-800 px-6 py-3">
        <h2 className="text-white font-medium text-sm">Chat</h2>
        <p className="text-gray-500 text-xs">BGE-M3 retrieval · Llama 3.2 answers</p>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-5">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} msg={msg} />
        ))}
        {loading && <TypingIndicator />}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-gray-800 px-6 py-4">
        <div className="flex gap-3 items-end">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                send()
              }
            }}
            placeholder="Ask a question about your documents…"
            rows={1}
            className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-2.5 text-sm text-gray-200 placeholder-gray-500 resize-none focus:outline-none focus:border-indigo-500 transition-colors"
            style={{ maxHeight: 120 }}
          />
          <button
            onClick={send}
            disabled={!input.trim() || loading}
            className="w-9 h-9 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center transition-colors flex-shrink-0"
          >
            <Send size={15} />
          </button>
        </div>
        <p className="text-xs text-gray-600 mt-1.5">Enter to send · Shift+Enter for new line</p>
      </div>
    </div>
  )
}
