/**
 * GEAS Dashboard - Interactive Sortable & Paginated Data Table with CSV Export
 */

export class DataTable {
  constructor({ containerId, title, columns, data = [], initialSort = null, pageSize = 15, onRowClick = null, prioritizeEligible = false, customComparator = null }) {
    this.container = typeof containerId === 'string' ? document.getElementById(containerId) : containerId;
    this.title = title;
    this.columns = columns; // Array of { key, label, format, sortable, align, tooltip }
    this.rawData = [...data];
    this.filteredData = [...data];
    this.sortKey = initialSort?.key || null;
    this.sortDir = initialSort?.dir || 'asc';
    this.currentPage = 1;
    this.pageSize = pageSize;
    this.searchQuery = '';
    this.onRowClick = onRowClick;
    this.prioritizeEligible = prioritizeEligible;
    this.customComparator = customComparator;

    this.init();
  }

  init() {
    if (!this.container) return;
    this.renderSkeleton();
    this.bindEvents();
    this.applySortAndFilter();
  }

  setData(newData) {
    this.rawData = [...newData];
    this.applySortAndFilter();
  }

  renderSkeleton() {
    this.container.innerHTML = `
      <div class="data-table-card">
        <div class="data-table-toolbar">
          <div class="table-toolbar-left">
            <h3 class="table-title">
              <span>${this.title}</span>
              <span class="table-row-count" data-el="count">0 rows</span>
            </h3>
          </div>
          <div class="table-toolbar-right">
            <input type="text" class="table-search-input" placeholder="Search table..." data-el="search">
            <select class="filter-select" style="width: auto; padding: 0.4rem 0.6rem; font-size: 0.8rem;" data-el="pagesize">
              <option value="10">10 / page</option>
              <option value="15" selected>15 / page</option>
              <option value="25">25 / page</option>
              <option value="50">50 / page</option>
              <option value="1000">All</option>
            </select>
            <button type="button" class="table-btn" data-el="export-csv">
              <span>📥</span> Export CSV
            </button>
          </div>
        </div>
        <div class="table-responsive-wrapper">
          <table class="data-table">
            <thead>
              <tr data-el="thead-row"></tr>
            </thead>
            <tbody data-el="tbody"></tbody>
          </table>
        </div>
        <div class="table-pagination-bar">
          <div data-el="page-info">Showing 0 to 0 of 0 entries</div>
          <div class="pagination-controls" data-el="pagination-btns"></div>
        </div>
      </div>
    `;

    // Render Table Header
    const theadRow = this.container.querySelector('[data-el="thead-row"]');
    theadRow.innerHTML = this.columns.map(col => {
      const isSortable = col.sortable !== false;
      const align = col.align || 'left';
      return `
        <th data-col-key="${col.key}" style="text-align: ${align}; cursor: ${isSortable ? 'pointer' : 'default'};" ${col.tooltip ? `title="${col.tooltip}"` : ''}>
          <span>${col.label}</span>
          ${isSortable ? `<span class="sort-indicator" data-sort-icon="${col.key}">↕</span>` : ''}
        </th>
      `;
    }).join('');
  }

  bindEvents() {
    // Search input
    const searchInput = this.container.querySelector('[data-el="search"]');
    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        this.searchQuery = e.target.value.toLowerCase().trim();
        this.currentPage = 1;
        this.applySortAndFilter();
      });
    }

    // Page size select
    const pageSizeSelect = this.container.querySelector('[data-el="pagesize"]');
    if (pageSizeSelect) {
      pageSizeSelect.addEventListener('change', (e) => {
        this.pageSize = parseInt(e.target.value, 10);
        this.currentPage = 1;
        this.renderBodyAndPagination();
      });
    }

    // Export CSV
    const exportBtn = this.container.querySelector('[data-el="export-csv"]');
    if (exportBtn) {
      exportBtn.addEventListener('click', () => this.exportCsv());
    }

    // Column Header Sorting
    const theadRow = this.container.querySelector('[data-el="thead-row"]');
    if (theadRow) {
      theadRow.addEventListener('click', (e) => {
        const th = e.target.closest('th');
        if (!th) return;
        const key = th.getAttribute('data-col-key');
        const colDef = this.columns.find(c => c.key === key);
        if (!colDef || colDef.sortable === false) return;

        if (this.sortKey === key) {
          this.sortDir = this.sortDir === 'asc' ? 'desc' : 'asc';
        } else {
          this.sortKey = key;
          this.sortDir = 'asc';
        }
        this.applySortAndFilter();
      });
    }
  }

  applySortAndFilter() {
    let list = [...this.rawData];

    // Filter by text search
    if (this.searchQuery) {
      list = list.filter(row => {
        return this.columns.some(col => {
          const val = row[col.key];
          if (val === null || val === undefined) return false;
          return String(val).toLowerCase().includes(this.searchQuery);
        });
      });
    }

    // Sorting
    if (this.customComparator) {
      list.sort(this.customComparator);
    } else if (this.sortKey) {
      list.sort((a, b) => {
        // Two-tier eligibility prioritization if enabled
        if (this.prioritizeEligible) {
          const aElig = Boolean(a.deployment_eligible && !a.baseline_only && a.model !== 'persistence');
          const bElig = Boolean(b.deployment_eligible && !b.baseline_only && b.model !== 'persistence');
          if (aElig !== bElig) return aElig ? -1 : 1;
        }

        let valA = a[this.sortKey];
        let valB = b[this.sortKey];

        // Handle nulls
        if (valA === null || valA === undefined) return 1;
        if (valB === null || valB === undefined) return -1;

        // Number comparison
        if (typeof valA === 'number' && typeof valB === 'number') {
          const numDiff = this.sortDir === 'asc' ? valA - valB : valB - valA;
          if (Math.abs(numDiff) > 1e-6) return numDiff;
          // Tie-break: prefer without_quality_flags
          if (a.feature_variant !== b.feature_variant) {
            return a.feature_variant === 'without_quality_flags' ? -1 : 1;
          }
          return 0;
        }

        // Boolean comparison
        if (typeof valA === 'boolean' && typeof valB === 'boolean') {
          return this.sortDir === 'asc' ? (valA === valB ? 0 : valA ? -1 : 1) : (valA === valB ? 0 : valA ? 1 : -1);
        }

        // String comparison
        const strA = String(valA).toLowerCase();
        const strB = String(valB).toLowerCase();
        if (strA < strB) return this.sortDir === 'asc' ? -1 : 1;
        if (strA > strB) return this.sortDir === 'asc' ? 1 : -1;
        return 0;
      });
    }

    this.filteredData = list;
    this.updateSortIcons();
    this.renderBodyAndPagination();
  }

  updateSortIcons() {
    const icons = this.container.querySelectorAll('[data-sort-icon]');
    icons.forEach(icon => {
      const key = icon.getAttribute('data-sort-icon');
      if (key === this.sortKey) {
        icon.textContent = this.sortDir === 'asc' ? '▲' : '▼';
        icon.style.opacity = '1';
      } else {
        icon.textContent = '↕';
        icon.style.opacity = '0.35';
      }
    });
  }

  renderBodyAndPagination() {
    const tbody = this.container.querySelector('[data-el="tbody"]');
    const countEl = this.container.querySelector('[data-el="count"]');
    const pageInfo = this.container.querySelector('[data-el="page-info"]');
    const paginationBtns = this.container.querySelector('[data-el="pagination-btns"]');

    const total = this.filteredData.length;
    countEl.textContent = `${total} row${total !== 1 ? 's' : ''}`;

    if (total === 0) {
      tbody.innerHTML = `<tr><td colspan="${this.columns.length}" style="text-align: center; padding: 2.5rem; color: var(--text-muted);">No matching records found.</td></tr>`;
      pageInfo.textContent = 'Showing 0 to 0 of 0 entries';
      paginationBtns.innerHTML = '';
      return;
    }

    const totalPages = Math.ceil(total / this.pageSize);
    if (this.currentPage > totalPages) this.currentPage = totalPages;
    if (this.currentPage < 1) this.currentPage = 1;

    const startIdx = (this.currentPage - 1) * this.pageSize;
    const endIdx = Math.min(startIdx + this.pageSize, total);
    const pageRows = this.filteredData.slice(startIdx, endIdx);

    // Render Table Rows
    tbody.innerHTML = pageRows.map(row => {
      const isIneligible = row.deployment_eligible === false || row.deployment_eligible === 'False' || row.deployment_eligible === 'false';
      const isBaseline = row.baseline_only === true || row.baseline_only === 'True' || row.model === 'persistence';
      const rowClass = isBaseline ? 'row-baseline' : isIneligible ? 'row-ineligible' : '';

      const cells = this.columns.map(col => {
        const align = col.align || 'left';
        let cellContent = row[col.key];
        if (col.format && typeof col.format === 'function') {
          cellContent = col.format(row[col.key], row);
        } else if (cellContent === null || cellContent === undefined) {
          cellContent = '<span style="color: var(--text-muted);">N/A</span>';
        }
        return `<td style="text-align: ${align};">${cellContent}</td>`;
      }).join('');

      return `<tr class="${rowClass}">${cells}</tr>`;
    }).join('');

    // Pagination info & buttons
    pageInfo.textContent = `Showing ${startIdx + 1} to ${endIdx} of ${total} entries`;

    let btnsHtml = '';
    btnsHtml += `<button type="button" class="page-btn" data-page="prev" ${this.currentPage === 1 ? 'disabled' : ''}>&laquo; Prev</button>`;

    const maxVisiblePages = 5;
    let startPage = Math.max(1, this.currentPage - 2);
    let endPage = Math.min(totalPages, startPage + maxVisiblePages - 1);
    if (endPage - startPage < maxVisiblePages - 1) {
      startPage = Math.max(1, endPage - maxVisiblePages + 1);
    }

    for (let p = startPage; p <= endPage; p++) {
      btnsHtml += `<button type="button" class="page-btn ${p === this.currentPage ? 'active' : ''}" data-page="${p}">${p}</button>`;
    }

    btnsHtml += `<button type="button" class="page-btn" data-page="next" ${this.currentPage === totalPages ? 'disabled' : ''}>Next &raquo;</button>`;
    paginationBtns.innerHTML = btnsHtml;

    // Bind page buttons
    paginationBtns.querySelectorAll('[data-page]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const action = e.target.getAttribute('data-page');
        if (action === 'prev') {
          if (this.currentPage > 1) this.currentPage--;
        } else if (action === 'next') {
          if (this.currentPage < totalPages) this.currentPage++;
        } else {
          this.currentPage = parseInt(action, 10);
        }
        this.renderBodyAndPagination();
      });
    });
  }

  exportCsv() {
    if (this.filteredData.length === 0) return;
    const headerRow = this.columns.map(c => `"${c.label.replace(/"/g, '""')}"`).join(',');
    const dataRows = this.filteredData.map(row => {
      return this.columns.map(col => {
        let val = row[col.key];
        if (val === null || val === undefined) val = '';
        if (typeof val === 'object') val = JSON.stringify(val);
        return `"${String(val).replace(/"/g, '""')}"`;
      }).join(',');
    });

    const csvContent = 'data:text/csv;charset=utf-8,\uFEFF' + [headerRow, ...dataRows].join('\r\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    const sanitizeTitle = this.title.toLowerCase().replace(/[^a-z0-9]/g, '_');
    link.setAttribute('download', `${sanitizeTitle}_export.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }
}
