window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: "^((?!arithmatex).)*$",
    processHtmlClass: "arithmatex",
  },
};

if (document.querySelector(".arithmatex")) {
  const script = document.createElement("script");
  script.src = "https://unpkg.com/mathjax@3.2.2/es5/tex-mml-chtml.js";
  script.async = true;
  document.head.append(script);
}
