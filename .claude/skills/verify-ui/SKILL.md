---
name: verify-ui
context: fork
description: >
  Frontend testing and web UI verification via DOM inspection in a Playwright browser. Verify component behavior, check layout, and test interactive elements after implementing or modifying features. Checks visibility, clickability, alignment, overflow, and state transitions — without screenshots. Use when verifying UI, investigating bugs, or testing acceptance criteria against a live app. Use when user reports a bug or asks to investigate how a feature works. Use BEFORE custom Playwright usage.
version: '1.0.0'
---

# Verify UI

Follow this process for every verification. Use the bundled scripts — do not write 
custom page.evaluate code unless the scripts cannot answer the question.

## Step 1: Understand what to verify
Read the minimum source code to identify:
1. How to reach the feature (route, buttons to click)
2. Key selectors (CSS classes, roles, test-ids)
3. Expected layout rules from CSS (flex direction, gaps, alignment)
Stop reading after these three. Do NOT explore the full component tree.
4. NEVER use ref for snapshots. Target by css class, role, or whatever else
here what happens when you try to get page snapshot by ref:
```console
⏺ playwright - Page snapshot (MCP)(target: "[ref=e31]", depth: 6)
  ⎿  Error: ### Error
     Error: "[ref=e31]" does not match any elements.
```
it happens all the time, please do not waste tokens on that

## Step 2: Navigate to the feature
- Use page.locator('selector').click() for all interactions
- Never take page snapshots
- Never use ref targets (see Step 1)

## Step 3: Load and run verification scripts
For EVERY measurement, follow this exact sequence:

### After-action state checks (THIS IS THE MOST COMMON CHECK YOU SHOULD USE IN MOST CASES)
This is a 3-stage check: setup observer, perform click with Playwright, collect results.
1. Read: `Read .claude/skills/verify-ui/scripts/check-after-action.js`
2. Inject into page: `page.evaluate(scriptContent)`
3. Setup: `page.evaluate((opts) => setupAfterAction(opts), { scope: '[role="dialog"]', expect: [{ selector: '.success-message', visible: true }, { selector: '.loading-spinner', visible: false }], timeout: 5000 })`
4. Click with Playwright: `page.locator('.submit-button').click()`
5. Collect: `page.evaluate(() => collectAfterAction())`
6. Check overall `pass` and each result's `reason`

Use this for: selection state changes, popup open/close, tab switching,
form submission results — any click that should change the UI permanently.
Use checkVisibilityTransition instead when tracking temporary elements
(spinners, toasts, animations).

### Alignment checks
1. Read the script: `Read .claude/skills/verify-ui/scripts/check-alignment.js`
2. Inject into page: `page.evaluate(scriptContent)` where scriptContent is the 
   full file you just read
3. Call: `page.evaluate(() => checkAlignment({ containerSelector: '...', scopeSelector: '...' }))`
4. Interpret the result: check `behavior`, `alignment`, `gaps.uniform`, `gaps.value`

### Clickability checks  
1. Read: `Read .claude/skills/verify-ui/scripts/check-clickable.js`
2. Inject into page
3. Call: `page.evaluate(() => checkClickable({ selectors: [...], scope: '...' }))`
4. Check each result's `clickable` and `reason`

### Visibility transition checks
This is a 3-stage check: setup observer, perform click with Playwright, collect results.
1. Read: `Read .claude/skills/verify-ui/scripts/check-visibility-transition.js`
2. Inject into page: `page.evaluate(scriptContent)`
3. Setup: `page.evaluate((opts) => setupVisibilityTransition(opts), { target: '.spinner', scope: '[role="dialog"]', expected: [false, true, false], timeout: 5000 })`
4. Click with Playwright: `page.locator('.submit-button').click()`
5. Collect: `page.evaluate(() => collectVisibilityTransition())`
6. Check `matches` and `reason`

### Container inventory checks
Use this after opening a popover, dialog, or dropdown to discover everything inside in one call
instead of writing custom discovery code each time.
1. Read: `Read .claude/skills/verify-ui/scripts/check-inventory.js`
2. Inject into page
3. Call: `page.evaluate(() => checkInventory({
     scope: '[data-example]', // e.g.: [data-popover-content], [data-radix-popper-content-wrapper], [role="dialog"]
     include: ['button', '[role="option"]', '[role="radio"]', '[role="group"]']
   }))`
4. Inspect `summary` for counts, `sections` for labeled groups, `scrollState` for scroll info,
   and `elements` for the full list with text/role/ariaState/visible/inViewport per element.

Options:
- `scope` (required): CSS selector for the container to inventory
- `include`: array of selectors to search for (defaults to all common interactive roles)
- `exclude`: array of selectors to skip
- `maxDepth`: limit DOM traversal depth

### Overflow and reachability checks
Use this to check whether a container or its children overflow the viewport,
whether footer content is clipped, and whether interactive elements are still reachable.
1. Read: `Read .claude/skills/verify-ui/scripts/check-overflow.js`
2. Inject into page
3. Call: `page.evaluate(() => checkOverflow({
     scope: '[data-side]',
     checkFooter: true
   }))`
4. Check `summary` for the quick verdict (`overflows`, `scrollable`, `footerClipped`,
   `unreachableCount`), `overflow`/`overflowPx` for directional detail, `footer` for
   last-child clipping, and `unreachable` for any interactive elements that can't be clicked.

Options:
- `scope` (required): CSS selector for the container to check
- `checkFooter`: check if the last visible child is clipped (default: true)
- `checkInteractive`: hit-test all interactive elements for reachability (default: true)
- `margin`: pixel tolerance for overflow detection (default: 0)

### How to inject scripts
Scripts live on disk, NOT accessible from the browser. To inject:
1. Use the Read tool: `Read .claude/skills/verify-ui/scripts/check-alignment.js`
2. The file content is now in YOUR context (not the browser's)
3. Inject via: `page.evaluate(fileContent)` — this defines the function in browser scope
4. Then call: `page.evaluate(() => checkAlignment({...}))`
NEVER use fetch() or import() to load scripts — the browser cannot access local files

## Step 4: Compile results
Populate all check in a md format table, with the columns "Elements", "Check", "Result", "Details", "Suggested fix" for instance:
```
| Elements | Check | Result | Details | Suggested fix |
|----------|-------|--------|---------|---------------|
| .search-input | visible | ✅ | Visible at the very top on the search form | N/A |
| .search-input | clickable | ✅ | elementFromPoint returns self | N/A |
| .dropdown | appears on focus | ❌ | MutationObserver: element never became visible within 3s | Add visibility state change on search input focus |
```
### Important Notes Towards Details And Summaries
Instead of providing numeric information, explain what is going on semantically. Imagine that user does not see the skill at all and you have to explain to him being in complete blindness

## General rules
- For clicking elements, ALWAYS use Playwright's native click via page.locator('selector').click() — never use dispatchEvent or element.click() from page.evaluate. Components often require real pointer events. 
- For selector, prioritize text which element contains or accessibility attributes (e.g. aria-role)
- To make selection more precise, instead of selecting element on the whole page, first extract feature box element from page (use text feature contains, classes, test-ids, consider mixing those approaches if you struggle to select single item, example: .locator('\[role="form"\]:has-text("Register Account")')), then use locator from feature element instead of page
- For tab switching, radio selection, checkbox filling, dropdown element picking and all other user interactions use native clicks: .locator(\<selector\>).click()
- use .fill() for text input and .press() or page.keyboard.press() for key presses, do not edit DOM value field directly
- avoid gathering getComputedStyle information other than about visibility/opacity: that is redundant, getBoundingClientRect provides full data about shape and positioning

## Custom scripts
Use custom skill ONLY when you tried to utilize scripts provided above and you still lack some information/low on confidence

## Navigation efficiency
- After reading source code, you already know the selectors. Do NOT take page snapshots
  just to "see" the page — go directly to clicking and measuring.
- Only use page snapshot when you genuinely don't know what's on the page
  (e.g., investigating an unknown bug with no code context).
- When you need to inspect a specific area of the page, use page.locator('css-selector').evaluate()
- NEVER use ref targets for Playwright, neither for snapshots nor for clicks

## browser_snapshot rules
  - The `target` parameter MUST be a CSS selector or role query, NEVER a [ref=...] value
  - you MUST map snapshot which contains ref values to source code, to transform ref to css class, text or whatever else
  - If you need a subtree snapshot, use a CSS selector like '[role="listbox"]' or '.className'
  - If you don't know the CSS selector, DON'T take a targeted snapshot — use page.evaluate with getBoundingClientRect or DOM queries instead

## What skill does not do
- Skill does not verify visual appearance (colors, fonts, spacing and contrast aesthetics), it verifies only if the feature elements behave as it planned by spec and codebase
- Skill does not test cross-browser behavior and uses only Playwright Chromium browser
- Skill does not verify accessibility features use different skill for that
- Skill avoids using screenshots and fallbacks to them only in case of low confidence after usage of tools
