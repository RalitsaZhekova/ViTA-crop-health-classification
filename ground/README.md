# Ground component

`ground/` is now the receiving side of the mission. Crop-condition calculations
run on the payload; the ground component will store compact downlink products,
build historical series, expose the API and serve the professional web client.

The authoritative scientific implementation remains available from the shared
package so payload and ground validation use exactly the same formulas. Ground
processing must not silently recalculate a different headline score.

## Ground responsibilities

- validate the downlink manifest and asset checksums;
- index scenes by region, footprint and acquisition time;
- serve the RGB preview and condition/quality overlay;
- expose exact measurements, explanations and evidence quality from JSON;
- compare compatible observations through time;
- produce alerts and client-facing history without claiming disease diagnosis.

Single-scene labels remain spectral screening priorities. `Nominal`, `Watch`,
`Moderate anomaly`, `High anomaly` and `Insufficient data` describe the available
multispectral evidence, not confirmed agronomic health or a specific cause.

The API, database adapter and web application are the next ground deliverables.
