// Keep horizontally scrollable display equations reachable by keyboard.
export default function accessibleMath() {
  return function transform(tree) {
    function visit(node) {
      if (node.type === 'element' && node.properties?.className?.includes('katex-display')) {
        node.properties.tabIndex = 0;
        node.properties.role = 'region';
        node.properties.ariaLabel = 'Mathematical equation';
      }
      for (const child of node.children ?? []) visit(child);
    }
    visit(tree);
  };
}
