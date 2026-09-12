import { useCallback, useEffect, useRef, useState } from 'react'

import { useAppStore, type Session } from '../../../stores/appStore'
import { chatOwnerKey } from '../../../utils/chatOwnership'

export interface GateContinuationPending {
  ownerKey: string
  action: string
  generation: number
  stateVersion: number
}

interface GateActionToken {
  ownerKey: string
  sessionId: string
  action: string
  generation: number
  stateVersion: number
}

interface GateActionSubmissionOptions {
  currentSession: Session | null
  stateVersion?: number
  wsConnect: (() => void) | null
  onGateActionError?: (error: unknown, action: string) => void
}

// A post-redrive state must be authoritative before it can settle a pending
// gate action. `paused` is explicitly excluded: the redrive protocol parks
// there transiently, and treating it as settlement is what exposed a stale
// Resume control.
const SETTLING_STATUSES = new Set([
  'running',
  'waiting_input',
  'waiting_annotation',
  'waiting_plan_review',
  'waiting_species_input',
  'generating_report',
  'completed',
  'failed',
  'stopped',
])

/** Owns the single-flight token and settlement rules shared by all gate posts. */
export function useGateActionSubmission(options: GateActionSubmissionOptions) {
  const [gateContinuationPending, setGateContinuationPending] =
    useState<GateContinuationPending | null>(null)
  // Visible pending and in-flight authority are deliberately separate. A
  // newer same-owner state may settle the visible indicator before the HTTP
  // promise does, but the matching promise must still be allowed to close its
  // dialog once it settles.
  const visiblePendingRef = useRef<GateContinuationPending | null>(null)
  const inFlightRef = useRef<GateActionToken | null>(null)
  const generationRef = useRef(0)
  const wsConnectRef = useRef(options.wsConnect)
  wsConnectRef.current = options.wsConnect

  const clearVisiblePending = useCallback((
    ownerKey?: string,
    generation?: number,
  ) => {
    const pending = visiblePendingRef.current
    if (!pending) return
    if (ownerKey !== undefined && pending.ownerKey !== ownerKey) return
    if (generation !== undefined && pending.generation !== generation) return
    visiblePendingRef.current = null
    setGateContinuationPending(null)
  }, [])

  const clearInFlight = useCallback((ownerKey?: string, generation?: number) => {
    const token = inFlightRef.current
    if (!token) return
    if (ownerKey !== undefined && token.ownerKey !== ownerKey) return
    if (generation !== undefined && token.generation !== generation) return
    inFlightRef.current = null
  }, [])

  // Selecting a different owner invalidates both the visible indicator and the
  // stale action token. A strictly newer authoritative state version settles
  // only the visible indicator for the owner that submitted the action.
  // PAUSED never settles anything.
  useEffect(() => {
    const token = inFlightRef.current
    const pending = visiblePendingRef.current
    const currentOwner = chatOwnerKey(
      options.currentSession?.id, options.currentSession?.restoreBranchId,
    )
    if (token && currentOwner !== token.ownerKey) {
      clearInFlight(token.ownerKey, token.generation)
      clearVisiblePending(token.ownerKey, token.generation)
      return
    }
    if (pending && currentOwner !== pending.ownerKey) {
      clearVisiblePending(pending.ownerKey, pending.generation)
      return
    }
    if (!pending && !token) return
    const status = String(options.currentSession?.status ?? '')
    if (status === 'paused') return
    const version = Number(options.stateVersion)
    if (!Number.isFinite(version)) return
    const observedVersion = token?.stateVersion ?? pending?.stateVersion ?? 0
    if (version <= observedVersion) return
    if (!SETTLING_STATUSES.has(status)) return
    clearVisiblePending(
      pending?.ownerKey ?? token?.ownerKey,
      pending?.generation ?? token?.generation,
    )
  }, [
    options.currentSession?.id,
    options.currentSession?.restoreBranchId,
    options.currentSession?.status,
    options.stateVersion,
    clearInFlight,
    clearVisiblePending,
  ])

  const beginGateAction = useCallback((action: string): GateActionToken | null => {
    const ownerKey = chatOwnerKey(
      options.currentSession?.id, options.currentSession?.restoreBranchId,
    )
    if (!ownerKey || !options.currentSession) return null
    const current = useAppStore.getState().currentSession
    if (chatOwnerKey(current?.id, current?.restoreBranchId) !== ownerKey) return null
    const existing = inFlightRef.current
    if (existing?.ownerKey === ownerKey) return null
    if (existing) {
      clearInFlight(existing.ownerKey, existing.generation)
      clearVisiblePending(existing.ownerKey, existing.generation)
    }
    const generation = generationRef.current + 1
    generationRef.current = generation
    const version = Number(options.stateVersion)
    const token: GateActionToken = {
      ownerKey,
      sessionId: options.currentSession.id,
      action,
      generation,
      stateVersion: Number.isFinite(version) ? version : 0,
    }
    const pending: GateContinuationPending = {
      ownerKey,
      action,
      generation,
      stateVersion: token.stateVersion,
    }
    inFlightRef.current = token
    visiblePendingRef.current = pending
    setGateContinuationPending(pending)
    return token
  }, [options.currentSession, options.stateVersion, clearInFlight, clearVisiblePending])

  const ownsInFlight = useCallback((ownerKey: string, generation: number) => {
    const token = inFlightRef.current
    return Boolean(
      token
      && token.ownerKey === ownerKey
      && token.generation === generation,
    )
  }, [])

  const reconcileAfterAwait = useCallback((ownerKey: string, generation: number) => {
    const current = useAppStore.getState().currentSession
    return (
      ownsInFlight(ownerKey, generation)
      && chatOwnerKey(current?.id, current?.restoreBranchId) === ownerKey
    )
  }, [ownsInFlight])

  const reconnectIfNeeded = useCallback((ownerKey: string) => {
    if (useAppStore.getState().wsConnected) return
    setTimeout(() => {
      const current = useAppStore.getState().currentSession
      if (
        chatOwnerKey(current?.id, current?.restoreBranchId) === ownerKey
        && !useAppStore.getState().wsConnected
      ) wsConnectRef.current?.()
    }, 100)
  }, [])

  const submitGateAction = useCallback(async (
    action: string,
    request: (token: GateActionToken) => Promise<unknown>,
    onSuccess: (ownerKey: string, generation: number) => void,
  ) => {
    const token = beginGateAction(action)
    if (!token) return false
    try {
      await request(token)
      if (!reconcileAfterAwait(token.ownerKey, token.generation)) return false
      // Clear authority only after the safe same-owner cleanup has run: a
      // visible-state settlement may already have hidden the indicator.
      clearInFlight(token.ownerKey, token.generation)
      onSuccess(token.ownerKey, token.generation)
      return true
    } catch (error) {
      // A failed old owner action is a no-op; its owner-change effect clears
      // both visible and in-flight state.
      if (!reconcileAfterAwait(token.ownerKey, token.generation)) return false
      clearInFlight(token.ownerKey, token.generation)
      clearVisiblePending(token.ownerKey, token.generation)
      options.onGateActionError?.(error, action)
      console.error(`Gate action failed (${action}):`, error)
      throw error
    }
  }, [
    beginGateAction, clearInFlight, clearVisiblePending, reconcileAfterAwait,
    options.onGateActionError,
  ])

  return { gateContinuationPending, reconnectIfNeeded, submitGateAction }
}
