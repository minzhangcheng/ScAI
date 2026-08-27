import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import ts from 'typescript'
import { describe, expect, it } from 'vitest'


const MAX_FILE_LINES = 1_000
const MAX_FUNCTION_LINES = 300
const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..')
const FRONTEND_ROOT = path.join(REPO_ROOT, 'frontend', 'src')

const BASELINE_LONG_TYPESCRIPT = new Set([
  'frontend/src/App.tsx::App',
  'frontend/src/components/analysis/FigureGallery.tsx::FigureGallery',
  'frontend/src/components/data/DataSection.tsx::DataSection',
  'frontend/src/pages/Admin.tsx::Admin',
])

const BASELINE_OVERSIZED_TYPESCRIPT = new Set([])

type FunctionLike = ts.FunctionDeclaration | ts.FunctionExpression | ts.ArrowFunction
  | ts.MethodDeclaration | ts.GetAccessorDeclaration | ts.SetAccessorDeclaration

function isProductionSource(filePath: string): boolean {
  const name = path.basename(filePath)
  return (
    /\.(?:ts|tsx|js|jsx)$/.test(name)
    && !name.endsWith('.d.ts')
    && !name.includes('.test.')
    && !name.includes('.spec.')
    && name !== 'test-setup.ts'
    && name !== 'test-setup.tsx'
    && !filePath.split(path.sep).includes('__tests__')
  )
}

function sourceFiles(directory: string): string[] {
  return fs.readdirSync(directory, { withFileTypes: true })
    .flatMap(entry => {
      const candidate = path.join(directory, entry.name)
      return entry.isDirectory() ? sourceFiles(candidate) : [candidate]
    })
    .filter(isProductionSource)
    .sort()
}

function relativePath(filePath: string): string {
  return path.relative(REPO_ROOT, filePath).split(path.sep).join('/')
}

function propertyName(node: ts.Node): string | null {
  if (ts.isIdentifier(node) || ts.isPrivateIdentifier(node)) return node.text
  if (ts.isStringLiteral(node) || ts.isNumericLiteral(node)) return node.text
  if (ts.isPropertyAccessExpression(node)) return node.getText()
  if (ts.isElementAccessExpression(node)) return node.getText()
  return null
}

function assignedName(node: FunctionLike): string | null {
  if ((ts.isFunctionDeclaration(node) || ts.isFunctionExpression(node)) && node.name) {
    return node.name.text
  }
  if (
    ts.isMethodDeclaration(node)
    || ts.isGetAccessorDeclaration(node)
    || ts.isSetAccessorDeclaration(node)
  ) {
    return propertyName(node.name)
  }

  let current: ts.Node = node
  while (current.parent && (
    ts.isParenthesizedExpression(current.parent)
    || ts.isAsExpression(current.parent)
    || ts.isSatisfiesExpression(current.parent)
  )) {
    current = current.parent
  }

  if (ts.isVariableDeclaration(current.parent)) {
    return propertyName(current.parent.name)
  }
  if (ts.isPropertyAssignment(current.parent)) {
    return propertyName(current.parent.name)
  }
  if (
    ts.isBinaryExpression(current.parent)
    && current.parent.right === current
    && current.parent.operatorToken.kind === ts.SyntaxKind.EqualsToken
  ) {
    return propertyName(current.parent.left)
  }
  if (ts.isCallExpression(current.parent)) {
    let owner: ts.Node = current.parent
    while (owner.parent && (
      ts.isParenthesizedExpression(owner.parent)
      || ts.isAsExpression(owner.parent)
      || ts.isSatisfiesExpression(owner.parent)
    )) {
      owner = owner.parent
    }
    if (ts.isVariableDeclaration(owner.parent)) {
      return propertyName(owner.parent.name)
    }
    if (
      ts.isBinaryExpression(owner.parent)
      && owner.parent.right === owner
      && owner.parent.operatorToken.kind === ts.SyntaxKind.EqualsToken
    ) {
      return propertyName(owner.parent.left)
    }
  }
  return null
}

function isFunctionLike(node: ts.Node): node is FunctionLike {
  return ts.isFunctionDeclaration(node)
    || ts.isFunctionExpression(node)
    || ts.isArrowFunction(node)
    || ts.isMethodDeclaration(node)
    || ts.isGetAccessorDeclaration(node)
    || ts.isSetAccessorDeclaration(node)
}

function functionPath(node: FunctionLike): string | null {
  const names: string[] = []
  let current: ts.Node | undefined = node
  while (current) {
    if (isFunctionLike(current)) {
      const name = assignedName(current)
      if (name && names[0] !== name) names.unshift(name)
    }
    current = current.parent
  }
  return names.length ? names.join('/') : null
}

function parseSource(filePath: string): ts.SourceFile {
  const kind = filePath.endsWith('.tsx') || filePath.endsWith('.jsx')
    ? ts.ScriptKind.TSX
    : ts.ScriptKind.TS
  return ts.createSourceFile(
    filePath,
    fs.readFileSync(filePath, 'utf8'),
    ts.ScriptTarget.Latest,
    true,
    kind,
  )
}

function scanLongFunctions(): Set<string> {
  const debt = new Set<string>()
  for (const filePath of sourceFiles(FRONTEND_ROOT)) {
    const source = parseSource(filePath)
    const visit = (node: ts.Node) => {
      if (isFunctionLike(node)) {
        const first = source.getLineAndCharacterOfPosition(node.getStart(source)).line
        const last = source.getLineAndCharacterOfPosition(node.end).line
        if (last - first + 1 > MAX_FUNCTION_LINES) {
          const name = functionPath(node)
          if (name) debt.add(`${relativePath(filePath)}::${name}`)
        }
      }
      ts.forEachChild(node, visit)
    }
    visit(source)
  }
  return debt
}

function scanOversizedFiles(): Set<string> {
  return new Set(
    sourceFiles(FRONTEND_ROOT)
      .filter(filePath => {
        const source = fs.readFileSync(filePath, 'utf8')
        const lines = source.length === 0
          ? 0
          : source.split(/\r\n|\r|\n/).length - (/\r\n$|\r$|\n$/.test(source) ? 1 : 0)
        return lines > MAX_FILE_LINES
      })
      .map(relativePath),
  )
}

describe('production TypeScript size debt', () => {
  it('matches the frozen planning-base inventory', () => {
    expect(scanLongFunctions()).toEqual(BASELINE_LONG_TYPESCRIPT)
    expect(scanOversizedFiles()).toEqual(BASELINE_OVERSIZED_TYPESCRIPT)
  })
})
