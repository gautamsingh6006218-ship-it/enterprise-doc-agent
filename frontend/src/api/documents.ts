import { apiClient } from './client'
import type { BatchIngestResponse, DeleteResponse, DocumentListResponse } from '../types/api'

export async function listDocuments(
  status?: string,
  limit = 50,
  offset = 0
): Promise<DocumentListResponse> {
  const params: Record<string, unknown> = { limit, offset }
  if (status) params.status = status
  const { data } = await apiClient.get<DocumentListResponse>('/documents', { params })
  return data
}

export async function deleteDocument(id: string): Promise<DeleteResponse> {
  const { data } = await apiClient.delete<DeleteResponse>(`/documents/${id}`)
  return data
}

export async function uploadFiles(
  files: File[],
  visibility = 'public',
  accessRoles = ''
): Promise<BatchIngestResponse> {
  const form = new FormData()
  files.forEach((f) => form.append('files', f))
  form.append('visibility', visibility)
  form.append('access_roles', accessRoles)
  const { data } = await apiClient.post<BatchIngestResponse>('/ingest/batch', form)
  return data
}

export async function uploadSingleFile(
  file: File,
  visibility = 'public'
): Promise<import('../types/api').IngestResponse> {
  const form = new FormData()
  form.append('file', file)
  form.append('visibility', visibility)
  const { data } = await apiClient.post('/ingest', form)
  return data
}
