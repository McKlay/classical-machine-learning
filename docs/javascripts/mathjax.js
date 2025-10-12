// Configure MathJax to accept $$...$$ and \[...\] display math, and \(...\) inline math.
window.MathJax = {
  tex: {
    inlineMath: [['\\(', '\\)']],
    displayMath: [['\\[', '\\]'], ['$$', '$$']],
    processEscapes: true,
    processEnvs: true
  },
  options: {
    // Don't skip common tags like div/span so pymdownx can mark math regions
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
  }
};

// Dynamically load MathJax v3 from CDN so the site will render math at runtime.
(function () {
  if (typeof window.MathJax !== 'undefined' && window.MathJax.startup && window.MathJax.startup.defaultReady) {
    return; // MathJax already present
  }

  var script = document.createElement('script');
  script.type = 'text/javascript';
  script.id = 'MathJax-script';
  script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
  script.async = true;
  document.head.appendChild(script);
})();
