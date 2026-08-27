import { useCallback, useEffect, useRef } from 'react'

import { getSnapshotTransaction } from '../../hooks/useApi'
import { useAppStore } from '../../stores/appStore'
import type { SnapshotTransactionRecord } from '../../types/snapshotV2'
import type { SnapshotRetryPollState } from './types'

const SNAPSHOT_RETRY_POLL_INTERVAL_MS = 2_000
const SNAPSHOT_RETRY_MAX_POLLS = 150
const SNAPSHOT_TERMINAL_STATUSES = new Set<SnapshotTransactionRecord['status']>([
  'committed', 'failed', 'cancelled', 'timed_out',
])

interface SnapshotRetryMonitorOptions {
  currentSessionId: string | null
  snapshotV2Enabled: boolean
  snapshotTransaction: SnapshotTransactionRecord | null
  setShowSnapshotPanel: (show: boolean) => void
}

export function useSnapshotRetryMonitor(options: SnapshotRetryMonitorOptions) {
  const snapshotRetryPollRef = useRef<SnapshotRetryPollState | null>(null)
  const snapshotAutoOpenedForRef = useRef<string | null>(null)

  const stopSnapshotRetryPolling = useCallback(() => {
    const poll = snapshotRetryPollRef.current
    if (poll) {
      poll.cancelled = true
      if (poll.timer) clearTimeout(poll.timer)
    }
    snapshotRetryPollRef.current = null
  }, [])

  const monitorSnapshotRetry = useCallback((sessionId: string, seed: SnapshotTransactionRecord) => {
    const activePoll = snapshotRetryPollRef.current
    if (activePoll?.transactionId === seed.transaction_id && !activePoll.cancelled) return
    stopSnapshotRetryPolling()
    useAppStore.getState().setSnapshotTransaction(seed)
    const pollState: SnapshotRetryPollState = {
      transactionId: seed.transaction_id,
      cancelled: false,
      attempts: 0,
      timer: null,
    }
    snapshotRetryPollRef.current = pollState

    const poll = async () => {
      if (pollState.cancelled) return
      const current = useAppStore.getState().snapshotTransaction
      if (current?.transaction_id !== pollState.transactionId) {
        stopSnapshotRetryPolling()
        return
      }
      if (SNAPSHOT_TERMINAL_STATUSES.has(current.status)) {
        stopSnapshotRetryPolling()
        return
      }
      if (pollState.attempts >= SNAPSHOT_RETRY_MAX_POLLS) {
        console.warn(
          'Lost contact with restore transaction', pollState.transactionId,
          '— stopped polling after', SNAPSHOT_RETRY_MAX_POLLS, 'attempts',
        )
        const stalled = useAppStore.getState().snapshotTransaction
        if (stalled && stalled.transaction_id === pollState.transactionId) {
          useAppStore.getState().setSnapshotTransaction({
            ...stalled,
            poll_timed_out: true,
          } as SnapshotTransactionRecord)
        }
        stopSnapshotRetryPolling()
        return
      }
      pollState.attempts += 1
      try {
        const transaction = await getSnapshotTransaction(sessionId, seed.transaction_id)
        if (pollState.cancelled) return
        const latest = useAppStore.getState().snapshotTransaction
        if (latest?.transaction_id !== pollState.transactionId) {
          stopSnapshotRetryPolling()
          return
        }
        useAppStore.getState().setSnapshotTransaction(transaction)
        if (SNAPSHOT_TERMINAL_STATUSES.has(transaction.status)) {
          stopSnapshotRetryPolling()
          return
        }
      } catch (error) {
        if ((error as { code?: string } | null)?.code === 'SNAPSHOT_V2_DISABLED') {
          useAppStore.getState().setSnapshotV2Enabled(false)
          stopSnapshotRetryPolling()
          return
        }
        console.error('Failed to poll snapshot retry transaction:', error)
      }
      if (!pollState.cancelled) {
        pollState.timer = setTimeout(poll, SNAPSHOT_RETRY_POLL_INTERVAL_MS)
      }
    }
    void poll()
  }, [stopSnapshotRetryPolling])

  const transactionId = options.snapshotTransaction?.transaction_id ?? null
  const transactionSessionId = options.snapshotTransaction?.session_id ?? null
  const transactionIsRetry = Boolean(options.snapshotTransaction?.retry_of_transaction_id)
  const transactionTerminal = options.snapshotTransaction
    ? SNAPSHOT_TERMINAL_STATUSES.has(options.snapshotTransaction.status)
    : true

  useEffect(() => {
    snapshotAutoOpenedForRef.current = null
    return stopSnapshotRetryPolling
  }, [stopSnapshotRetryPolling, options.currentSessionId])

  useEffect(() => {
    if (!options.snapshotV2Enabled || !options.currentSessionId || !transactionId) return
    if (transactionSessionId !== options.currentSessionId) return
    if (transactionTerminal) {
      stopSnapshotRetryPolling()
      return
    }
    const seed = useAppStore.getState().snapshotTransaction
    if (!seed || seed.transaction_id !== transactionId) return
    monitorSnapshotRetry(options.currentSessionId, seed)
  }, [
    options.currentSessionId,
    options.snapshotV2Enabled,
    transactionId,
    transactionSessionId,
    transactionTerminal,
    monitorSnapshotRetry,
    stopSnapshotRetryPolling,
  ])

  useEffect(() => {
    if (!options.snapshotV2Enabled || !options.currentSessionId || !transactionId) return
    if (transactionSessionId !== options.currentSessionId || !transactionIsRetry || transactionTerminal) return
    if (snapshotAutoOpenedForRef.current === transactionId) return
    snapshotAutoOpenedForRef.current = transactionId
    options.setShowSnapshotPanel(true)
  }, [
    options.currentSessionId,
    transactionId,
    transactionSessionId,
    transactionIsRetry,
    transactionTerminal,
    options.snapshotV2Enabled,
  ])

  return { monitorSnapshotRetry, snapshotAutoOpenedForRef, stopSnapshotRetryPolling }
}
