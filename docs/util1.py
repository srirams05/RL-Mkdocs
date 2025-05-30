
# Script to convert \emph{text} to $\textit{text}$ in chapter1.md
# This script reads a Markdown file, replaces all instances of \emph{text} with $\textit{text}$,
# and writes the changes back to the same file.

import re

with open('chapter1.md', 'r', encoding='utf-8') as file:
    content = file.read()

# Replace \emph{text} with $\textit{text}$
updated_content = re.sub(r'\\emph{([^}]*)}', r'$\\textit{\1}$', content)

with open('chapter1.md', 'w', encoding='utf-8') as file:
    file.write(updated_content)