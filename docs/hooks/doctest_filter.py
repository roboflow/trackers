"""
MkDocs hook to strip doctest directives from rendered documentation.

Removes `# doctest: +SKIP`, `# doctest: +ELLIPSIS`, and similar directives
from code blocks so they don't appear in the rendered docs.
"""

import re


def on_page_content(html: str, **kwargs) -> str:
    """
    Process page HTML content to remove doctest directives.

    This hook runs after markdown is converted to HTML, so we need to
    handle HTML-encoded content within <code> blocks.
    """
    # Pattern to match doctest directives in code
    # Handles both plain text and HTML-encoded versions
    patterns = [
        # Plain text version: # doctest: +DIRECTIVE
        r'\s*#\s*doctest:\s*\+\w+',
        # HTML-encoded version: # doctest: +DIRECTIVE
        r'\s*#\s*doctest:\s*\+\w+',
    ]

    for pattern in patterns:
        html = re.sub(pattern, '', html)

    return html
