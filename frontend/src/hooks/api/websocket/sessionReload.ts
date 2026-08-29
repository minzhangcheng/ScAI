import { useAppStore } from '../../../stores/appStore'
import { transformLoadResponse } from '../../../utils/transformLoadResponse'
import { apiFetch, applyStoreUpdates } from '../client'

export function reloadSessionState(
  sessionId: string,
  errorLabel: string,
  ownsResponse: () => boolean,
): Promise<any | null> {
  const sessionAtRequest = useAppStore.getState().currentSession
  if (sessionAtRequest?.id !== sessionId) return Promise.resolve(null)
  const versionAtRequest = useAppStore.getState().stateVersion
  const ownsRequest = () => (
    ownsResponse()
    && useAppStore.getState().currentSession === sessionAtRequest
  )
  return apiFetch<any>(
    `/session/${sessionId}/load`,
    undefined,
    ownsRequest,
  )
    .then(reloadData => {
      if (!reloadData || typeof reloadData !== 'object') return null
      if (
        !ownsRequest()
        || useAppStore.getState().currentSession?.id !== sessionId
      ) return null
      const currentVersion = useAppStore.getState().stateVersion
      const responseVersion = typeof reloadData.state_version === 'number'
        ? reloadData.state_version
        : null
      const storeAdvancedDuringRequest = currentVersion > versionAtRequest
      if (
        storeAdvancedDuringRequest
        && (responseVersion === null || responseVersion < currentVersion)
      ) return null
      const { sessionUpdates, storeUpdates } = transformLoadResponse(reloadData)
      useAppStore.getState().updateSession(sessionUpdates)
      applyStoreUpdates(storeUpdates)
      return reloadData
    })
    .catch(reloadErr => {
      if (ownsRequest()) console.error(errorLabel, reloadErr)
      return null
    })
}
