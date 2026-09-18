# Room Visualizer SDK

Framework-independent JavaScript/TypeScript SDK responsible for creating and
managing the embedded iframe widget on customer websites.

**Status:** SDK + Widget milestone — real implementation.

## Usage

```html
<div id="room-visualizer"></div>
<script type="module">
  import RoomVisualizer from "@room-visualizer/sdk";

  const widget = RoomVisualizer.init({
    tenantId: "a1b2c3d4-e5f6-47a8-89b0-1234567890ab", // PUBLIC tenant id only
    container: "#room-visualizer",
  });

  // Later, if needed:
  widget.destroy();
</script>
```

## Security

`tenantId` is the only tenant-identifying value this SDK ever accepts,
stores, or transmits. It is public information — never pass a secret
API key here. There is no configuration field for one.

All `postMessage` traffic to/from the iframe is validated against the
configured `widgetOrigin` before being trusted, and the iframe is
sandboxed to the minimum permissions the widget needs
(`allow-scripts allow-forms allow-same-origin`).

## Development

```bash
npm install
npm run build   # type-check + compile to dist/
npm test        # vitest + jsdom
```
