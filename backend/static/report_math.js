// Render protected report formulas without network access or trusted TeX commands.
document.querySelectorAll('[data-scai-math]').forEach(function (element) {
  katex.render(element.getAttribute('data-scai-math'), element, {
    displayMode: element.getAttribute('data-display') === 'true',
    output: 'mathml',
    trust: false,
    throwOnError: false,
  });
});
