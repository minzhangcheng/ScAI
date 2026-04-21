import { JWT_STORAGE_KEY } from '../constants/auth'

let _cachedToken: string | null | undefined = undefined

export function getToken(): string | null {
  if (_cachedToken === undefined) {
    _cachedToken = localStorage.getItem(JWT_STORAGE_KEY)
  }
  return _cachedToken
}

export function setToken(token: string): void {
  localStorage.setItem(JWT_STORAGE_KEY, token)
  _cachedToken = token
}

export function clearToken(): void {
  localStorage.removeItem(JWT_STORAGE_KEY)
  _cachedToken = null
}