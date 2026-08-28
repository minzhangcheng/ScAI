import { useAppStore } from '../../../stores/appStore'
import { transformLoadResponse } from '../../../utils/transformLoadResponse'
import { apiFetch, applyStoreUpdates } from '../client'

export function reloadSessionState(
  sessionId: string,
  errorLabel: string,
  ownsResponse: () => boolean,
): Promise<boolean> {
  return apiFetch<any>(
    `/session/${sessionId}/load`,
    undefined,
    ownsResponse,
  )
    .then(reloadData => {
      if (!reloadData || typeof reloadData !== 'object') return false
      if (
        !ownsResponse()
        || useAppStore.getState().currentSession?.id !== sessionId
      ) return false
      const { sessionUpdates, storeUpdates } = transformLoadResponse(reloadData)
      useAppStore.getState().updateSession(sessionUpdates)
      applyStoreUpdates(storeUpdates)
      return true
    })
    .catch(reloadErr => {
      if (ownsResponse()) console.error(errorLabel, reloadErr)
      return false
    })
}
