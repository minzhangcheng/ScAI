import { useEffect, useRef, useState } from 'react'

import { API_BASE } from '../../constants/api'
import { useAppStore, type PublicationFigure, type Session } from '../../stores/appStore'
import { chatOwnerKey } from '../../utils/chatOwnership'
import { fetchAndDownload, FetchError } from '../../utils/download'
import { reportBranchQuery } from '../../utils/reportBranch'

interface DownloadUiState {
  ownerKey: string | null
  ownerEpoch: number
  requestGeneration: number
  downloading: string | null
  downloadComplete: string | null
  downloadError: string | null
}

function emptyDownloadState(
  ownerKey: string | null,
  ownerEpoch: number,
): DownloadUiState {
  return {
    ownerKey,
    ownerEpoch,
    requestGeneration: 0,
    downloading: null,
    downloadComplete: null,
    downloadError: null,
  }
}

/** Download feedback and requests are scoped to one session/restore owner. */
export function useOwnedDownloads(
  currentSession: Session | null,
  figures: PublicationFigure[],
) {
  const branchQuery = reportBranchQuery(currentSession?.restoreBranchId)
  const ownerKey = chatOwnerKey(
    currentSession?.id, currentSession?.restoreBranchId,
  )
  const ownerEpochRef = useRef(0)
  const previousOwnerRef = useRef(ownerKey)
  if (previousOwnerRef.current !== ownerKey) {
    previousOwnerRef.current = ownerKey
    ownerEpochRef.current += 1
  }
  const ownerEpoch = ownerEpochRef.current
  const requestGenerationRef = useRef(0)
  const [downloadState, setDownloadState] = useState<DownloadUiState>(
    () => emptyDownloadState(ownerKey, 0),
  )
  const visibleState = downloadState.ownerKey === ownerKey
    && downloadState.ownerEpoch === ownerEpoch
    ? downloadState
    : emptyDownloadState(ownerKey, ownerEpoch)

  useEffect(() => {
    setDownloadState(previous => (
      previous.ownerKey === ownerKey && previous.ownerEpoch === ownerEpoch
        ? previous
        : emptyDownloadState(ownerKey, ownerEpoch)
    ))
  }, [ownerKey, ownerEpoch])

  const ownsRenderedOwner = () => {
    const active = useAppStore.getState().currentSession
    return Boolean(
      currentSession
      && ownerKey
      && chatOwnerKey(active?.id, active?.restoreBranchId) === ownerKey,
    )
  }

  const ownsRequest = (
    requestOwner: string,
    requestOwnerEpoch: number,
    generation: number,
  ) => (
    ownsRenderedOwner()
    && ownerKey === requestOwner
    && ownerEpochRef.current === requestOwnerEpoch
    && requestGenerationRef.current === generation
  )

  const showSuccess = (type: string, requestOwner: string, generation: number) => {
    if (!ownsRequest(requestOwner, ownerEpoch, generation)) return
    setDownloadState(previous => (
      previous.ownerKey === requestOwner
        && previous.ownerEpoch === ownerEpoch
        && previous.requestGeneration === generation
        ? { ...previous, downloadComplete: type }
        : previous
    ))
    setTimeout(() => {
      if (!ownsRequest(requestOwner, ownerEpoch, generation)) return
      setDownloadState(previous => (
        previous.ownerKey === requestOwner
          && previous.ownerEpoch === ownerEpoch
          && previous.requestGeneration === generation
          ? { ...previous, downloadComplete: null }
          : previous
      ))
    }, 2000)
  }

  const showError = (message: string, requestOwner: string, generation: number) => {
    if (!ownsRequest(requestOwner, ownerEpoch, generation)) return
    setDownloadState(previous => (
      previous.ownerKey === requestOwner
        && previous.ownerEpoch === ownerEpoch
        && previous.requestGeneration === generation
        ? { ...previous, downloadError: message }
        : previous
    ))
    // Preserve the previous self-clearing error behavior without allowing an
    // old owner's timeout to mutate the currently rendered owner.
    setTimeout(() => {
      if (!ownsRequest(requestOwner, ownerEpoch, generation)) return
      setDownloadState(previous => (
        previous.ownerKey === requestOwner
          && previous.ownerEpoch === ownerEpoch
          && previous.requestGeneration === generation
          ? { ...previous, downloadError: null }
          : previous
      ))
    }, 10000)
  }

  const downloadType = async (
    type: string,
    url: string,
    filename: string,
    options?: RequestInit,
  ) => {
    if (!currentSession || !ownerKey || !ownsRenderedOwner()) return
    const generation = requestGenerationRef.current + 1
    requestGenerationRef.current = generation
    setDownloadState({
      ownerKey,
      ownerEpoch,
      requestGeneration: generation,
      downloading: type,
      downloadComplete: null,
      downloadError: null,
    })
    try {
      await fetchAndDownload(url, filename, options)
      showSuccess(type, ownerKey, generation)
    } catch (e) {
      showError(
        e instanceof Error ? e.message : `Failed to download ${type}`,
        ownerKey,
        generation,
      )
    } finally {
      if (ownsRequest(ownerKey, ownerEpoch, generation)) {
        setDownloadState(previous => (
          previous.ownerKey === ownerKey
            && previous.ownerEpoch === ownerEpoch
            && previous.requestGeneration === generation
            ? { ...previous, downloading: null }
            : previous
        ))
      }
    }
  }

  const downloadAllFigures = () => {
    if (!currentSession || !ownsRenderedOwner()) return
    return downloadType(
      'figures',
      `${API_BASE}/session/${currentSession.id}/report/download-all-figures${branchQuery}`,
      `figures_${currentSession.id}.zip`,
      { method: 'POST' },
    )
  }

  const downloadAllTables = () => {
    if (!currentSession || !ownsRenderedOwner()) return
    return downloadType(
      'tables',
      `${API_BASE}/session/${currentSession.id}/report/download-all-tables${branchQuery}`,
      `tables_${currentSession.id}.zip`,
      { method: 'POST' },
    )
  }

  const downloadAll = async () => {
    if (!currentSession || !ownerKey || !ownsRenderedOwner()) return
    const requestOwner = ownerKey
    const generation = requestGenerationRef.current + 1
    requestGenerationRef.current = generation
    setDownloadState({
      ownerKey: requestOwner,
      ownerEpoch,
      requestGeneration: generation,
      downloading: 'all',
      downloadComplete: null,
      downloadError: null,
    })
    try {
      const downloads: Promise<void>[] = []
      if (figures.length > 0) {
        downloads.push(
          fetchAndDownload(
            `${API_BASE}/session/${currentSession.id}/report/download-all-figures${branchQuery}`,
            `figures_${currentSession.id}.zip`,
            { method: 'POST' },
          ),
        )
      }
      // Always attempt the tables zip — it includes score_definitions.csv even
      // when there are no publication tables. Ignore 404 (no tables directory).
      downloads.push(
        fetchAndDownload(
          `${API_BASE}/session/${currentSession.id}/report/download-all-tables${branchQuery}`,
          `tables_${currentSession.id}.zip`,
          { method: 'POST' },
        ).catch((e) => {
          if (e instanceof FetchError && e.status === 404) return
          throw e
        }),
      )
      await Promise.all(downloads)
      showSuccess('all', requestOwner, generation)
    } catch (e) {
      showError(
        e instanceof Error ? e.message : 'Failed to download',
        requestOwner,
        generation,
      )
    } finally {
      if (ownsRequest(requestOwner, ownerEpoch, generation)) {
        setDownloadState(previous => (
          previous.ownerKey === requestOwner
            && previous.ownerEpoch === ownerEpoch
            && previous.requestGeneration === generation
            ? { ...previous, downloading: null }
            : previous
        ))
      }
    }
  }

  const dismissDownloadError = () => setDownloadState(previous => (
    previous.ownerKey === ownerKey
      ? { ...previous, downloadError: null }
      : previous
  ))

  return {
    ...visibleState,
    dismissDownloadError,
    downloadAll,
    downloadAllFigures,
    downloadAllTables,
  }
}
