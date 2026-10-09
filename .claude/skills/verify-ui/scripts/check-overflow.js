// check-overflow.js
// Checks whether a container or any of its children extend beyond the viewport,
// whether it's scrollable, and whether all interactive elements inside remain reachable.
//
// Usage with Playwright MCP browser_evaluate:
//   1. Inject: browser_evaluate({ function: "<paste checkOverflow function>" })
//   2. Call:   browser_evaluate({ function: "() => checkOverflow({ scope: '[data-side]' })" })
//
// Or with browser_run_code_unsafe:
//   async (page) => page.evaluate((opts) => checkOverflow(opts), { scope: '...', checkFooter: true })

function checkOverflow({ scope, checkFooter, checkInteractive, margin } = {}) {
  if (!scope) {
    return { error: 'Provide a "scope" selector' };
  }

  const scopeEl = document.querySelector(scope);
  if (!scopeEl) {
    return { error: `Scope element not found: ${scope}` };
  }

  const viewport = { width: window.innerWidth, height: window.innerHeight };
  const scopeRect = scopeEl.getBoundingClientRect();
  const edgeMargin = margin || 0;

  // --- Container overflow ---
  const overflow = {
    top: scopeRect.top < (0 - edgeMargin),
    right: scopeRect.right > (viewport.width + edgeMargin),
    bottom: scopeRect.bottom > (viewport.height + edgeMargin),
    left: scopeRect.left < (0 - edgeMargin),
  };

  const overflows = overflow.top || overflow.right || overflow.bottom || overflow.left;

  const overflowPx = {
    top: overflow.top ? Math.round(0 - scopeRect.top) : 0,
    right: overflow.right ? Math.round(scopeRect.right - viewport.width) : 0,
    bottom: overflow.bottom ? Math.round(scopeRect.bottom - viewport.height) : 0,
    left: overflow.left ? Math.round(0 - scopeRect.left) : 0,
  };

  // --- Scrollability of scope and its children ---
  function getScrollInfo(el) {
    const cs = getComputedStyle(el);
    const overflowY = cs.overflowY;
    const overflowX = cs.overflowX;
    const scrollableY =
      (overflowY === "auto" || overflowY === "scroll" || overflowY === "overlay") &&
      el.scrollHeight > el.clientHeight;
    const scrollableX =
      (overflowX === "auto" || overflowX === "scroll" || overflowX === "overlay") &&
      el.scrollWidth > el.clientWidth;

    if (!scrollableY && !scrollableX) return null;

    return {
      scrollableY,
      scrollableX,
      overflowY,
      overflowX,
      scrollHeight: el.scrollHeight,
      clientHeight: el.clientHeight,
      scrollWidth: el.scrollWidth,
      clientWidth: el.clientWidth,
      hiddenPxY: scrollableY ? el.scrollHeight - el.clientHeight : 0,
      hiddenPxX: scrollableX ? el.scrollWidth - el.clientWidth : 0,
    };
  }

  const scopeScroll = getScrollInfo(scopeEl);

  // Check direct children for scrollable containers
  let scrollableChildren = [];
  for (const child of scopeEl.children) {
    const info = getScrollInfo(child);
    if (info) {
      const tag = child.tagName.toLowerCase();
      const cls =
        child.className && typeof child.className === "string"
          ? "." + child.className.split(" ").filter(Boolean).slice(0, 2).join(".")
          : "";
      scrollableChildren.push({
        selector: tag + cls,
        ...info,
      });
    }
  }

  const scrollable = !!(scopeScroll || scrollableChildren.length);

  // --- Max-height / constraint detection ---
  const cs = getComputedStyle(scopeEl);
  const constraints = {
    maxHeight: cs.maxHeight !== "none" ? cs.maxHeight : null,
    maxWidth: cs.maxWidth !== "none" ? cs.maxWidth : null,
    overflow: cs.overflow,
    overflowY: cs.overflowY,
    overflowX: cs.overflowX,
  };

  // --- Child overflow scan ---
  // Find children that overflow the scope container's bounds
  const clippedChildren = [];
  const scopeBottom = scopeRect.bottom;
  const scopeRight = scopeRect.right;

  function scanChildren(el, depth) {
    if (depth > 5) return; // limit depth
    for (const child of el.children) {
      const rect = child.getBoundingClientRect();
      if (rect.width === 0 && rect.height === 0) continue;

      const clipped = {
        bottom: rect.bottom > Math.min(scopeBottom, viewport.height) + edgeMargin,
        right: rect.right > Math.min(scopeRight, viewport.width) + edgeMargin,
        top: rect.top < Math.max(scopeRect.top, 0) - edgeMargin,
        left: rect.left < Math.max(scopeRect.left, 0) - edgeMargin,
      };

      if (clipped.bottom || clipped.right || clipped.top || clipped.left) {
        const tag = child.tagName.toLowerCase();
        const text = (child.textContent || "").trim().substring(0, 50);
        const role = child.getAttribute("role");
        clippedChildren.push({
          selector: buildSelector(child),
          tag,
          role,
          text: text || null,
          clipped,
          rect: {
            top: Math.round(rect.top),
            left: Math.round(rect.left),
            bottom: Math.round(rect.bottom),
            right: Math.round(rect.right),
            width: Math.round(rect.width),
            height: Math.round(rect.height),
          },
        });
      }

      scanChildren(child, depth + 1);
    }
  }

  scanChildren(scopeEl, 0);

  // --- Footer check ---
  let footer = null;
  if (checkFooter !== false) {
    // Find the last meaningful child (skip empty text nodes)
    const children = Array.from(scopeEl.children).filter((c) => {
      const r = c.getBoundingClientRect();
      return r.width > 0 && r.height > 0;
    });

    if (children.length > 0) {
      const lastChild = children[children.length - 1];
      const lastRect = lastChild.getBoundingClientRect();
      const clippedBottom = lastRect.bottom > viewport.height + edgeMargin;
      const clippedRight = lastRect.right > viewport.width + edgeMargin;

      footer = {
        selector: buildSelector(lastChild),
        text: (lastChild.textContent || "").trim().substring(0, 80),
        clipped: clippedBottom || clippedRight,
        clippedBottom,
        clippedRight,
        rect: {
          top: Math.round(lastRect.top),
          bottom: Math.round(lastRect.bottom),
          height: Math.round(lastRect.height),
        },
        overflowPx: clippedBottom
          ? Math.round(lastRect.bottom - viewport.height)
          : 0,
      };
    }
  }

  // --- Interactive element reachability ---
  let unreachable = null;
  if (checkInteractive !== false) {
    const interactiveSelectors = [
      "button",
      "a[href]",
      "input",
      "select",
      "textarea",
      '[role="button"]',
      '[role="link"]',
      '[role="option"]',
      '[role="radio"]',
      '[role="checkbox"]',
      '[role="switch"]',
      '[role="tab"]',
      '[role="menuitem"]',
    ];

    const allInteractive = scopeEl.querySelectorAll(
      interactiveSelectors.join(",")
    );
    const unreachableList = [];

    for (const el of allInteractive) {
      const rect = el.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) continue;

      const elCs = getComputedStyle(el);
      if (elCs.display === "none" || elCs.visibility === "hidden" || elCs.opacity === "0") {
        unreachableList.push({
          selector: buildSelector(el),
          text: (el.textContent || "").trim().substring(0, 50),
          role: el.getAttribute("role") || el.tagName.toLowerCase(),
          reason: "invisible",
          reachable: false,
        });
        continue;
      }
      if (
        elCs.pointerEvents === "none" ||
        el.disabled === true ||
        el.getAttribute("aria-disabled") === "true"
      ) {
        unreachableList.push({
          selector: buildSelector(el),
          text: (el.textContent || "").trim().substring(0, 50),
          role: el.getAttribute("role") || el.tagName.toLowerCase(),
          reason: "not interactive",
          reachable: false,
        });
        continue;
      }

      const issue = testReachability(el, rect);
      if (issue) {
        unreachableList.push({
          selector: buildSelector(el),
          text: (el.textContent || "").trim().substring(0, 50),
          role: el.getAttribute("role") || el.tagName.toLowerCase(),
          ...issue,
        });
      }
    }

    if (unreachableList.length > 0) {
      unreachable = unreachableList;
    }
  }

  function testReachability(el, rect) {
    // Outside viewport entirely
    if (
      rect.bottom < 0 ||
      rect.top > viewport.height ||
      rect.right < 0 ||
      rect.left > viewport.width
    ) {
      return { reason: "outside viewport", reachable: false };
    }

    // Partially outside viewport (clipped)
    if (rect.bottom > viewport.height || rect.right > viewport.width) {
      // Check if center is still within viewport
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      if (cx > viewport.width || cy > viewport.height) {
        return {
          reason: "center outside viewport",
          reachable: false,
          clippedPx: {
            bottom: rect.bottom > viewport.height ? Math.round(rect.bottom - viewport.height) : 0,
            right: rect.right > viewport.width ? Math.round(rect.right - viewport.width) : 0,
          },
        };
      }
    }

    // Hit-test from center
    const cx = Math.min(rect.left + rect.width / 2, viewport.width - 1);
    const cy = Math.min(rect.top + rect.height / 2, viewport.height - 1);
    if (cx < 0 || cy < 0) return { reason: "center outside viewport", reachable: false };

    const topEl = document.elementFromPoint(cx, cy);
    if (!topEl) return { reason: "elementFromPoint returned null", reachable: false };
    if (el === topEl || el.contains(topEl))
      return null; // reachable

    const tag = topEl.tagName.toLowerCase();
    const cls =
      topEl.className && typeof topEl.className === "string"
        ? "." + CSS.escape(topEl.className.split(" ").filter(Boolean)[0] || "")
        : "";
    return {
      reason: "covered by another element",
      reachable: false,
      coveredBy: `${tag}${cls}`,
    };
  }

  function buildSelector(el) {
    if (el.id) return `#${CSS.escape(el.id)}`;
    const testId =
      el.getAttribute("data-testid") || el.getAttribute("data-test-id");
    if (testId) return `[data-testid="${CSS.escape(testId)}"]`;
    const tag = el.tagName.toLowerCase();
    const role = el.getAttribute("role");
    const ariaLabel = el.getAttribute("aria-label");
    const cls =
      el.className && typeof el.className === "string"
        ? "." + el.className.split(" ").filter(Boolean).slice(0, 2).map(c => CSS.escape(c)).join(".")
        : "";
    if (role && ariaLabel)
      return `${tag}[role="${CSS.escape(role)}"][aria-label="${CSS.escape(ariaLabel)}"]`;
    if (role) return `${tag}[role="${CSS.escape(role)}"]${cls}`;
    return `${tag}${cls}`;
  }

  // --- Summary ---
  return {
    scope,
    scopeRect: {
      top: Math.round(scopeRect.top),
      left: Math.round(scopeRect.left),
      width: Math.round(scopeRect.width),
      height: Math.round(scopeRect.height),
      bottom: Math.round(scopeRect.bottom),
      right: Math.round(scopeRect.right),
    },
    viewport,
    overflows,
    overflow,
    overflowPx,
    scrollable,
    scrollState: scopeScroll || null,
    scrollableChildren: scrollableChildren.length ? scrollableChildren : null,
    constraints,
    footer,
    clippedChildren: clippedChildren.length ? clippedChildren : null,
    unreachable,
    summary: {
      overflows,
      scrollable,
      footerClipped: footer ? footer.clipped : false,
      unreachableCount: unreachable ? unreachable.length : 0,
      clippedChildCount: clippedChildren.length,
    },
  };
}
