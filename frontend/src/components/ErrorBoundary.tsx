import React from 'react'

type State = { hasError: boolean }

export class ErrorBoundary extends React.Component<React.PropsWithChildren, State> {
  state: State = { hasError: false }

  static getDerivedStateFromError() {
    return { hasError: true }
  }

  componentDidCatch(err: unknown) {
    console.error('UI crash:', err)
  }

  render() {
    if (this.state.hasError) {
      return <div role="alert">Something went wrong. Please reload.</div>
    }
    return this.props.children
  }
}
