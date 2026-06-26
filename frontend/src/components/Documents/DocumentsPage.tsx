import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Trash2, RefreshCw, FileText, CheckCircle, AlertCircle, Copy } from 'lucide-react'
import { listDocuments, deleteDocument } from '../../api/documents'
import UploadZone from './UploadZone'
import type { DocumentRecord } from '../../types/api'

const STATUS_STYLES: Record<string, string> = {
  completed: 'bg-green-900/50 text-green-400 border border-green-800',
  duplicate: 'bg-yellow-900/50 text-yellow-400 border border-yellow-800',
  pending: 'bg-blue-900/50 text-blue-400 border border-blue-800',
  failed_ingestion: 'bg-red-900/50 text-red-400 border border-red-800',
  failed_preprocessing: 'bg-red-900/50 text-red-400 border border-red-800',
  failed_chunking: 'bg-red-900/50 text-red-400 border border-red-800',
  failed_embedding: 'bg-red-900/50 text-red-400 border border-red-800',
}

function DocumentCard({ doc, onDelete }: { doc: DocumentRecord; onDelete: (id: string) => void }) {
  const [confirmDelete, setConfirmDelete] = useState(false)
  const name = doc.file_path.split('/').pop() || doc.id
  const failed = doc.status.startsWith('failed')

  return (
    <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-3 min-w-0">
          <div className={`mt-0.5 flex-shrink-0 ${failed ? 'text-red-500' : 'text-gray-500'}`}>
            {failed ? <AlertCircle size={16} /> : <FileText size={16} />}
          </div>
          <div className="min-w-0">
            <p className="text-sm text-white font-medium truncate">{name}</p>
            <p className="text-xs text-gray-500 font-mono mt-0.5 truncate">{doc.id}</p>
          </div>
        </div>
        <span className={`text-xs px-2 py-0.5 rounded-full flex-shrink-0 ${STATUS_STYLES[doc.status] || 'bg-gray-700 text-gray-400'}`}>
          {doc.status.replace('failed_', '')}
        </span>
      </div>

      <div className="mt-3 flex items-center gap-4 text-xs text-gray-500">
        {doc.status === 'completed' && (
          <>
            <span className="flex items-center gap-1">
              <CheckCircle size={11} className="text-green-500" />
              {doc.total_chunks} chunks
            </span>
            <span>{doc.total_duration_ms.toFixed(0)}ms</span>
          </>
        )}
        {doc.is_duplicate && (
          <span className="flex items-center gap-1">
            <Copy size={11} />
            duplicate
          </span>
        )}
        {doc.error && <span className="text-red-400 truncate max-w-48">{doc.error}</span>}
        <span className="ml-auto">
          {doc.created_at ? new Date(doc.created_at).toLocaleDateString() : '—'}
        </span>
      </div>

      <div className="mt-3 flex justify-end">
        {confirmDelete ? (
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-400">Delete?</span>
            <button
              onClick={() => onDelete(doc.id)}
              className="text-xs bg-red-600 hover:bg-red-500 px-2 py-1 rounded text-white transition-colors"
            >
              Yes
            </button>
            <button
              onClick={() => setConfirmDelete(false)}
              className="text-xs bg-gray-700 hover:bg-gray-600 px-2 py-1 rounded transition-colors"
            >
              Cancel
            </button>
          </div>
        ) : (
          <button
            onClick={() => setConfirmDelete(true)}
            className="text-gray-600 hover:text-red-400 transition-colors"
          >
            <Trash2 size={14} />
          </button>
        )}
      </div>
    </div>
  )
}

export default function DocumentsPage() {
  const qc = useQueryClient()

  const { data, isLoading, refetch } = useQuery({
    queryKey: ['documents'],
    queryFn: () => listDocuments(undefined, 50),
  })

  const deleteMutation = useMutation({
    mutationFn: deleteDocument,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['documents'] }),
  })

  const docs = data?.documents ?? []
  const completed = docs.filter((d) => d.status === 'completed').length
  const failed = docs.filter((d) => d.status.startsWith('failed')).length

  return (
    <div className="h-screen overflow-y-auto">
      <div className="border-b border-gray-800 px-6 py-3 sticky top-0 bg-gray-950 z-10">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-white font-medium text-sm">Documents</h2>
            <p className="text-gray-500 text-xs">
              {docs.length} total · {completed} indexed · {failed} failed
            </p>
          </div>
          <button
            onClick={() => refetch()}
            className="text-gray-500 hover:text-gray-300 transition-colors"
          >
            <RefreshCw size={15} />
          </button>
        </div>
      </div>

      <div className="px-6 py-4 space-y-6">
        <div>
          <h3 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-3">Upload</h3>
          <UploadZone onUploaded={() => qc.invalidateQueries({ queryKey: ['documents'] })} />
        </div>

        <div>
          <h3 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-3">
            Ingested Documents
          </h3>

          {isLoading && (
            <div className="text-center py-8 text-gray-500 text-sm">Loading…</div>
          )}

          {!isLoading && docs.length === 0 && (
            <div className="text-center py-8 text-gray-600 text-sm">
              No documents yet. Upload a file to get started.
            </div>
          )}

          <div className="space-y-2">
            {docs.map((doc) => (
              <DocumentCard
                key={doc.id}
                doc={doc}
                onDelete={(id) => deleteMutation.mutate(id)}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
