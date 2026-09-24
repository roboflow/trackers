/**
 * Grouped tables, such as the ReID results: the first row of each group names it in the first
 * column and the rows below leave that cell empty. Merge each name into one cell spanning its
 * group, and mark where each group starts so the stylesheet can draw a rule above it.
 */
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll(".md-typeset table").forEach(function (table) {
    const body = table.tBodies[0];
    if (!body || body.rows.length === 0) return;
    const rows = Array.from(body.rows);
    const isBlank = (row) => row.cells[0] && row.cells[0].textContent.trim() === "";
    if (isBlank(rows[0]) || !rows.some(isBlank)) return;

    let nameCell = null;
    rows.forEach(function (row) {
      if (isBlank(row)) {
        nameCell.rowSpan += 1;
        row.cells[0].remove();
      } else {
        nameCell = row.cells[0];
        row.classList.add("group-start");
      }
    });
    table.setAttribute("data-grouped", "");
  });
});
