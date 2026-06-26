import { useCallback, useState } from 'react'
import { Upload, X, FileText, CheckCircle, AlertCircle, Loader2 } from 'lucide-react'
import { uploadFiles } from '../../api/documents'
import type { IngestResponse } from '../../types/api'

interface FileItem {
  file: File
  status: 'pending' | 'uploading' | 'done' | 'error'
  result?: IngestResponse
}

export default function UploadZone({ onUploaded }: { onUploaded: () => void }) {
  const [files, setFiles] = useState<FileItem[]>([])
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)

  const addFiles = (incoming: FileList | File[]) => {
    const arr = Array.from(incoming)
    setFiles((prev) => [
      ...prev,
      ...arr.map((f) => ({ file: f, status: 'pending' as const })),
    ])
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files)
  }, [])

  const onFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) addFiles(e.target.files)
    e.target.value = ''
  }

  const remove = (i: number) => setFiles((prev) => prev.filter((_, idx) => idx !== i))

  const upload = async () => {
    const pending = files.filter((f) => f.status === 'pending')
    if (!pending.length) return
    setUploading(true)
    setFiles((prev) => prev.map((f) => f.status === 'pending' ? { ...f, status: 'uploading' } : f))

    try {
      const res = await uploadFiles(pending.map((f) => f.file))
      setFiles((prev) =>
        prev.map((item) => {
          const resultIdx = pending.findIndex((p) => p.file === item.file)
          if (resultIdx === -1) return item
          const r = res.results[resultIdx]
          return { ...item, status: r?.success ? 'done' : 'error', result: r }
        })
      )
      onUploaded()
    } catch {
      setFiles((prev) => prev.map((f) => f.status === 'uploading' ? { ...f, status: 'error' } : f))
    } finally {
      setUploading(false)
    }
  }

  const pendingCount = files.filter((f) => f.status === 'pending').length

  return (
    <div className="space-y-3">
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
          dragging ? 'border-indigo-500 bg-indigo-500/10' : 'border-gray-700 hover:border-gray-600'
        }`}
      >
        <Upload size={24} className="mx-auto text-gray-500 mb-2" />
        <p className="text-sm text-gray-400">Drop files here or{' '}
          <label className="text-indigo-400 cursor-pointer hover:text-indigo-300">
            browse
            <input type="file" multiple className="hidden" onChange={onFileInput}
              accept=".pdf,.txt,.md,.html,.docx,.pptx,.xlsx,.csv,.eml,.png,.jpg,.jpeg" />
          </label>
        </p>
        <p className="text-xs text-gray-600 mt-1">PDF, DOCX, PPTX, XLSX, TXT, MD, HTML, images</p>
      </div>

      {files.length > 0 && (
        <div className="space-y-1.5">
          {files.map((item, i) => (
            <div key={i} className="flex items-center gap-3 bg-gray-800 rounded-lg px-3 py-2">
              <FileText size={14} className="text-gray-500 flex-shrink-0" />
              <span className="flex-1 text-xs text-gray-300 truncate">{item.file.name}</span>
              <span className="text-xs text-gray-500">
                {(item.file.size / 1024).toFixed(0)} KB
              </span>
              {item.status === 'uploading' && <Loader2 size={14} className="animate-spin text-indigo-400" />}
              {item.status === 'done' && (
                <span className="flex items-center gap-1">
                  {item.result?.is_duplicate
                    ? <span className="text-xs text-yellow-500">duplicate</span>
                    : <CheckCircle size={14} className="text-green-500" />}
                </span>
              )}
              {item.status === 'error' && <AlertCircle size={14} className="text-red-500" />}
              {item.status === 'pending' && (
                <button onClick={() => remove(i)} className="text-gray-600 hover:text-gray-400">
                  <X size={14} />
                </button>
              )}
            </div>
          ))}

          {pendingCount > 0 && (
            <button
              onClick={upload}
              disabled={uploading}
              className="w-full mt-1 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-sm font-medium transition-colors"
            >
              {uploading ? 'Uploading…' : `Upload ${pendingCount} file${pendingCount > 1 ? 's' : ''}`}
            </button>
          )}
        </div>
      )}
    </div>
  )
}
