import axios from 'axios'

const BASE_URL = '/api'

export const apiClient = axios.create({
  baseURL: BASE_URL,
})

// Attach JWT token from localStorage to every request
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('jwt_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// On 401: clear token and redirect to login
apiClient.interceptors.response.use(
  (res) => res,
  (error) => {
    if (error.response?.status === 401) {
      clearToken()
      if (!window.location.pathname.includes('/login')) {
        window.location.href = '/login'
      }
    }
    return Promise.reject(error)
  }
)

export const getToken = () => localStorage.getItem('jwt_token')
export const setToken = (token: string) => localStorage.setItem('jwt_token', token)
export const clearToken = () => localStorage.removeItem('jwt_token')
